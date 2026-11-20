#!/usr/bin/env python3
"""
Gemma-2-9B-IT inference (LoRA adapter or zero-shot base model) for the
Spider positive / negative Text-to-SQL test sets, with optional execution
accuracy.

Same options, prompt content (text2sql_prompt.py), batching, --resume,
execution accuracy, --paired_file analysis and summary as the Qwen and
Llama inference scripts. Gemma-specific parts:

- Gated model: accept the licence on Hugging Face and run
  `huggingface-cli login` once.
- No system role in Gemma 2's template: the rules open the first user turn,
  exactly as in 01_finetune_gemma.py.
- Generation stops at <end_of_turn> or <eos>; decoding is greedy.
- Attention is "eager", as recommended for Gemma 2 (soft-capping).

Example:
  python 02_inference_gemma.py --mode neg \
      --input_file final_datasets/neg_test.json \
      --paired_file final_datasets/pos_test.json \
      --tables_file data/raw/tables.json --database_dir data/raw/database \
      --adapter_dir adapters/gemma_neg --output_file results/gemma_neg_on_neg.jsonl
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import collections
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text2sql_prompt import build_messages, extract_sql, get_sql, load_schema_map  # noqa: E402


MODEL_NAME = "google/gemma-2-9b-it"

STOP_TOKENS = ["<end_of_turn>", "<eos>"]


# ------------------------------------------------------------
# ARGUMENTS
# ------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pos", "neg"], required=True,
                        help="Label of the test file (stored in the output).")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--tables_file", required=True)
    parser.add_argument("--output_file", required=True, help="JSON Lines output.")
    parser.add_argument("--adapter_dir", default=None,
                        help="LoRA adapter. Omit for the zero-shot base model.")
    parser.add_argument("--model_name", default=MODEL_NAME)

    parser.add_argument("--database_dir", default=None,
                        help="Spider database folder; enables execution accuracy.")
    parser.add_argument("--paired_file", default=None,
                        help="The other polarity's test file (same ids), to detect "
                             "predictions that return the paired answer.")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_input_length", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Skip ids already present in --output_file.")
    parser.add_argument("--no_schema_keys", action="store_true",
                        help="Schema without types/keys - must match training.")

    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--query_timeout", type=float, default=30.0)
    return parser.parse_args()


args = parse_args()

print("=" * 80)
print("Gemma-2-9B-IT Inference")
print("=" * 80)
for key, value in vars(args).items():
    print(f"{key:<22}: {value}")
print("=" * 80)


# ------------------------------------------------------------
# EXECUTION HELPERS
# ------------------------------------------------------------

def find_database(db_id):
    folder = os.path.join(args.database_dir, db_id)
    path = os.path.join(folder, f"{db_id}.sqlite")
    if os.path.exists(path):
        return path
    if os.path.isdir(folder):
        for name in os.listdir(folder):
            if name.endswith(".sqlite"):
                return os.path.join(folder, name)
    return None


def execute(db_id, sql):
    """(rows, error) with every value as a string, like the dataset answers."""
    path = find_database(db_id)
    if path is None:
        return None, "database_missing"
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.text_factory = lambda b: b.decode("utf-8", "ignore")
    started = time.time()
    connection.set_progress_handler(
        lambda: 1 if time.time() - started > args.query_timeout else 0, 10000)
    try:
        rows = connection.execute(sql).fetchall()
        return [[None if v is None else str(v) for v in row] for row in rows], None
    except Exception as exc:
        return None, str(exc)
    finally:
        connection.close()


def canonical_value(value):
    if value is None:
        return ""
    text = str(value).strip()
    try:
        number = float(text)
        return repr(round(number, 6)) if number != int(number) else str(int(number))
    except (ValueError, OverflowError):
        return text


def results_match(gold_rows, pred_rows, ordered):
    if pred_rows is None or gold_rows is None:
        return False
    gold = [tuple(canonical_value(v) for v in row) for row in gold_rows]
    pred = [tuple(canonical_value(v) for v in row) for row in pred_rows]
    return gold == pred if ordered else sorted(gold) == sorted(pred)


def has_order_by(sql):
    return bool(re.search(r"\border\s+by\b", sql or "", re.IGNORECASE))


# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

schema_map = load_schema_map(args.tables_file, with_types=not args.no_schema_keys,
                             with_keys=not args.no_schema_keys)
print(f"Loaded {len(schema_map)} databases.")

with open(args.input_file, "r", encoding="utf-8") as handle:
    data = json.load(handle)
if args.max_samples is not None:
    data = data[:args.max_samples]

paired = {}
if args.paired_file:
    with open(args.paired_file, "r", encoding="utf-8") as handle:
        paired = {row["id"]: row for row in json.load(handle)}

output_path = Path(args.output_file)
output_path.parent.mkdir(parents=True, exist_ok=True)

done_ids = set()
if args.resume and output_path.exists():
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                done_ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
todo = [row for row in data if row.get("id") not in done_ids]
print(f"Examples: {len(data)}   already done: {len(done_ids)}   to run: {len(todo)}")


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir or args.model_name,
                                          trust_remote_code=True)
vocab = tokenizer.get_vocab()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

eos_ids = [vocab[t] for t in STOP_TOKENS if t in vocab]
if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in eos_ids:
    eos_ids.append(tokenizer.eos_token_id)

print("\nLoading base model in 4-bit NF4...")
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ),
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    trust_remote_code=True,
)
if args.adapter_dir:
    from peft import PeftModel
    print(f"Loading LoRA adapter from {args.adapter_dir}...")
    model = PeftModel.from_pretrained(model, args.adapter_dir)
else:
    print("No adapter: zero-shot base model.")
model.eval()
device = next(model.parameters()).device

generation_kwargs = {
    "max_new_tokens": args.max_new_tokens,
    "do_sample": args.do_sample,
    "repetition_penalty": args.repetition_penalty,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": eos_ids,
    "use_cache": True,
}
if args.do_sample:
    generation_kwargs.update(temperature=args.temperature, top_p=args.top_p)
else:
    # make sure no sampling defaults apply: decoding is greedy
    generation_kwargs.update(temperature=None, top_p=None, top_k=None)


def gemma_messages(messages):
    """Gemma 2's chat template has no system role (it raises an error), so
    the rules are put at the top of the first user turn."""
    if messages and messages[0]["role"] == "system":
        system, rest = messages[0]["content"], [dict(m) for m in messages[1:]]
        rest[0]["content"] = system + "\n\n" + rest[0]["content"]
        return rest
    return messages


def prompt_for(example):
    return tokenizer.apply_chat_template(
        gemma_messages(build_messages(example, schema_map, with_answer=False)),
        tokenize=False, add_generation_prompt=True,
    )


def generate_batch(prompts):
    encoded = tokenizer(prompts, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(device)
    with torch.inference_mode():
        output = model.generate(**encoded, **generation_kwargs)
    new_tokens = output[:, encoded["input_ids"].shape[1]:]
    return [tokenizer.decode(t, skip_special_tokens=True) for t in new_tokens]


# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------

def finish_record(example, raw_output, predicted_sql, error=None):
    record = dict(example)
    record.update(
        predicted_sql=predicted_sql,
        raw_output=raw_output,
        model=args.model_name,
        adapter_dir=args.adapter_dir,
        mode=args.mode,
    )
    if error:
        record["error"] = error
    if args.database_dir and predicted_sql:
        gold_answer = example.get("answer")
        if gold_answer is None:
            gold_answer, _ = execute(example["db_id"], get_sql(example))
        pred_rows, pred_error = execute(example["db_id"], predicted_sql)
        ordered = has_order_by(get_sql(example))
        record["execution_error"] = pred_error
        record["execution_match"] = results_match(gold_answer, pred_rows, ordered)
        partner = paired.get(example.get("id"))
        if partner is not None:
            record["matches_paired_answer"] = (
                results_match(partner.get("answer"), pred_rows, ordered)
                and not record["execution_match"]
            )
    elif args.database_dir:
        record["execution_match"] = False
        record["execution_error"] = error or "empty_prediction"
    return record


print("\n" + "=" * 80)
print("STARTING INFERENCE")
print("=" * 80)

skipped_long = 0
with output_path.open("a" if args.resume else "w", encoding="utf-8") as out:
    batch = []

    def flush(batch):
        try:
            outputs = generate_batch([prompt for _, prompt in batch])
            errors = [None] * len(batch)
        except Exception as exc:  # e.g. OOM: fall back to one by one
            outputs, errors = [], []
            torch.cuda.empty_cache()
            for _, prompt in batch:
                try:
                    outputs.append(generate_batch([prompt])[0])
                    errors.append(None)
                except Exception as single_exc:
                    outputs.append("")
                    errors.append(str(single_exc) or str(exc))
        for (example, _), raw, error in zip(batch, outputs, errors):
            record = finish_record(example, raw, extract_sql(raw) if raw else "", error)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            mark = ""
            if "execution_match" in record:
                mark = "OK " if record["execution_match"] else "BAD"
            print(f"[{example.get('id')}] {mark} {record['predicted_sql']}")
        out.flush()

    for example in todo:
        prompt = prompt_for(example)
        if len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) > args.max_input_length:
            skipped_long += 1
            record = finish_record(example, "", "", error="prompt_longer_than_max_input_length")
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            continue
        batch.append((example, prompt))
        if len(batch) == args.batch_size:
            flush(batch)
            batch = []
    if batch:
        flush(batch)


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

with output_path.open("r", encoding="utf-8") as handle:
    records = [json.loads(line) for line in handle if line.strip()]

summary = {
    "input_file": args.input_file,
    "adapter_dir": args.adapter_dir,
    "mode": args.mode,
    "examples": len(records),
    "empty_predictions": sum(1 for r in records if not r.get("predicted_sql")),
    "skipped_long_prompts": skipped_long,
}

if args.database_dir:
    def accuracy(rows):
        return round(100.0 * sum(r.get("execution_match", False) for r in rows) / len(rows), 2) \
            if rows else None

    summary["execution_accuracy"] = accuracy(records)
    summary["execution_errors"] = sum(1 for r in records if r.get("execution_error"))
    if paired:
        hits = sum(1 for r in records if r.get("matches_paired_answer"))
        summary["returns_paired_answer"] = hits
        summary["returns_paired_answer_percent"] = round(100.0 * hits / len(records), 2)
    for field in ("negation_type", "question_rule", "source_dataset"):
        groups = collections.defaultdict(list)
        for record in records:
            if record.get(field):
                groups[record[field]].append(record)
        if groups:
            summary[f"accuracy_by_{field}"] = {
                key: {"n": len(rows), "accuracy": accuracy(rows)}
                for key, rows in sorted(groups.items())
            }

summary_path = output_path.with_suffix(".summary.json")
with summary_path.open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)

print("\n" + "=" * 80)
print("INFERENCE COMPLETE")
print("=" * 80)
for key in ("examples", "empty_predictions", "skipped_long_prompts",
            "execution_accuracy", "execution_errors", "returns_paired_answer_percent"):
    if key in summary:
        print(f"{key:<32}: {summary[key]}")
print(f"Predictions: {output_path}")
print(f"Summary    : {summary_path}")
print("=" * 80)