#!/usr/bin/env python3
"""
Few-shot Text-to-SQL with Gemma-2-9B-IT on the Spider positive / negative
test sets, with execution accuracy.

Same shot selection, iterations, prompt content, evaluation and outputs as
the Qwen and Llama few-shot scripts. Gemma-specific parts:

- google/gemma-2-9b-it is gated: accept the licence on Hugging Face and run
  `huggingface-cli login` once (or set HF_TOKEN).
- Gemma 2's chat template has no system role, so the rules open the first
  user turn (the same text the other models get as a system message).
  Demonstrations are earlier user/model turns.
- Generation stops at <end_of_turn> or <eos>.
- Attention is "eager", as recommended for Gemma 2 (logit soft-capping).
- Gemma 2 has an 8,192-token context window, so --max_input_tokens defaults
  to 7,680 (room for 256 new tokens), and --batch_size defaults to 1
  (eager attention on long few-shot prompts needs much memory).

What changed from the previous Gemma few-shot script (01_gemma.py):

- FIX: the SAME few-shot examples (5 random training questions) were used
  for every test question. Shots are now chosen per question by question
  similarity, exactly as for Qwen and Llama, and change across iterations
  (--num_iterations).
- FIX: the prompt was raw text without Gemma's chat template, while the
  other models used their chat templates. Gemma now gets its chat template
  and the same prompt content as the other models and the fine-tuned runs.
- FIX: generation SAMPLED (temperature 0.7, top_k 50, repetition_penalty
  1.05), so predictions changed randomly between runs and results were not
  reproducible. Decoding is greedy now; variation between iterations comes
  from the different demonstrations. --do_sample is still available.
- FIX: prompts were truncated from the RIGHT at 4,096 tokens. Five
  demonstrations with their schemas often exceed that, which cut off the
  test question itself. Nothing is truncated now: the least similar shots
  are dropped until the prompt fits, and the number used is recorded.
- FIX: in neg mode the prompt got an extra paragraph about negation, which
  tells the model which test set it is on. Both modes use the same prompt.
- FIX: generation stopped only at <eos>, not at Gemma's end of turn.
- Shot count: the old Gemma script used 5 shots while the Qwen/Llama
  scripts used 3. All three now default to 3; pass the same --num_shots
  (e.g. 5) to every model so results are comparable.
- Metric: string exact match under-counts correct SQL (aliases, quotes,
  column order). Execution accuracy against the gold answers in the test
  file is now the main metric (--database_dir); exact match is still
  reported. With --paired_file the script also counts predictions that
  return the PAIRED question's answer (ignored or invented negation).
- Same prompt as the fine-tuned models (text2sql_prompt.py: identical
  system rules, schema with types and keys; demonstrations as earlier
  user/assistant turns), so few-shot and fine-tuned results compare.
- Greedy decoding is really greedy: the model's sampling defaults are
  switched off and repetition_penalty is 1.0. --max_new_tokens is 256 (the longest
  gold SQL is ~150 tokens; 128 could cut it).
- Shots never come from the test question itself or its paraphrase group,
  even if the train and test files overlap by mistake. --exclude_same_db
  restricts shots to other databases: Spider has near-identical questions
  with slightly different SQL (e.g. the same question plus "in descending
  order") that the similarity search will otherwise find.
- Records keep the test example's fields (id, answer, negation_type, ...)
  plus the shot ids; --resume works on ids. A summary JSON is written.

- Iterations: --num_iterations N runs N iterations with the model loaded
  once. Each iteration shows DIFFERENT demonstrations: for every test
  question, shots are drawn from its --shot_pool_size most similar training
  examples (or the whole pool with --shot_strategy random), and an
  iteration never repeats a shot set already used for that question in an
  earlier iteration (as long as the pool has enough combinations). The
  draw depends only on --seed, the question id and the iteration number, so
  iteration 3 gives the same shots whether it runs in one job or alone
  (--start_iteration 3 --num_iterations 1). Each iteration writes
  <output>_iter<i>.jsonl and its summary; <output>.iterations_summary.json
  holds every iteration's metrics plus mean and standard deviation.

Example (negative test set, demonstrations from the negative training set):
  python 03_fewshot_gemma.py --mode neg \
      --train_file final_datasets/neg_train.json \
      --test_file final_datasets/neg_test.json \
      --paired_file final_datasets/pos_test.json \
      --tables_file data/raw/tables.json --database_dir data/raw/database \
      --num_iterations 5 \
      --output_file results/fewshot_gemma_neg_on_neg.jsonl
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import collections
import json
import math
import random
import re
import sqlite3
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text2sql_prompt import (  # noqa: E402
    build_messages, extract_sql, get_question, get_sql, load_schema_map, normalize_sql,
)


MODEL_NAME = "google/gemma-2-9b-it"

STOP_TOKENS = ["<end_of_turn>", "<eos>"]


# ------------------------------------------------------------
# ARGUMENTS
# ------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Few-shot Spider Text-to-SQL with Gemma-2-9B-IT")
    parser.add_argument("--mode", choices=["pos", "neg"], required=True,
                        help="Label of the test file (stored in the output).")
    parser.add_argument("--train_file", nargs="+", required=True,
                        help="Demonstration pool: one or more training JSON files.")
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--tables_file", required=True)
    parser.add_argument("--output_file", required=True, help="JSON Lines output.")
    parser.add_argument("--model_name", default=MODEL_NAME)

    parser.add_argument("--database_dir", default=None,
                        help="Spider database folder; enables execution accuracy.")
    parser.add_argument("--paired_file", default=None,
                        help="The other polarity's test file (same ids).")

    parser.add_argument("--num_shots", type=int, default=3)
    parser.add_argument("--num_iterations", type=int, default=1,
                        help="Number of iterations, each with different demonstrations.")
    parser.add_argument("--start_iteration", type=int, default=0,
                        help="First iteration number (to run iterations in separate jobs).")
    parser.add_argument("--shot_pool_size", type=int, default=None,
                        help="'similar': draw shots from this many most similar examples "
                             "(default 4 x num_shots). Equal to num_shots = always the same "
                             "top shots, so iterations would repeat.")
    parser.add_argument("--shot_strategy", choices=["similar", "random"], default="similar")
    parser.add_argument("--same_db_bonus", type=float, default=0.1,
                        help="Added to the similarity of shots from the test question's database.")
    parser.add_argument("--exclude_same_db", action="store_true",
                        help="Only use shots from OTHER databases (stricter setting: no "
                             "near-duplicate questions of the same database).")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_input_tokens", type=int, default=7680,
                        help="Gemma 2's context is 8,192 tokens including the answer.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Skip ids already present in each iteration's output file.")
    parser.add_argument("--no_schema_keys", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--query_timeout", type=float, default=30.0)
    return parser.parse_args()


args = parse_args()
if args.shot_pool_size is None:
    args.shot_pool_size = 4 * args.num_shots
if args.shot_pool_size < args.num_shots:
    raise ValueError("--shot_pool_size must be at least --num_shots.")
if args.num_iterations > 1 and args.shot_strategy == "similar" and args.shot_pool_size == args.num_shots:
    print("WARNING: shot_pool_size == num_shots, so every iteration uses the same shots.")
random.seed(args.seed)
torch.manual_seed(args.seed)

if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU detected; a 7B model is not practical on CPU.")

print("=" * 80)
print("Gemma-2-9B-IT Few-shot Spider Evaluation")
print("=" * 80)
print(f"{'cuda_device':<20}: {torch.cuda.get_device_name(0)}")
for key, value in vars(args).items():
    print(f"{key:<20}: {value}")
print("=" * 80)


# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

schema_map = load_schema_map(args.tables_file, with_types=not args.no_schema_keys,
                             with_keys=not args.no_schema_keys)

train_pool = []
for path in args.train_file:
    with open(path, "r", encoding="utf-8") as handle:
        rows = [r for r in json.load(handle) if r.get("db_id") in schema_map]
    train_pool.extend(rows)
    print(f"Demonstrations from {path}: {len(rows)}")

with open(args.test_file, "r", encoding="utf-8") as handle:
    test_data = json.load(handle)
if args.max_samples is not None:
    test_data = test_data[:args.max_samples]
print(f"Test examples: {len(test_data)}")

if len(train_pool) < args.num_shots:
    raise ValueError(f"{args.num_shots} shots requested but the pool has {len(train_pool)} examples.")

paired = {}
if args.paired_file:
    with open(args.paired_file, "r", encoding="utf-8") as handle:
        paired = {row["id"]: row for row in json.load(handle)}


# ------------------------------------------------------------
# SHOT SELECTION
# ------------------------------------------------------------

# Function words carry no topic; negation words are deliberately NOT here.
STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "by", "for", "to", "from", "with",
    "and", "or", "is", "are", "was", "were", "be", "been", "what", "which",
    "who", "whom", "whose", "how", "many", "much", "find", "show", "list",
    "give", "return", "display", "me", "all", "each", "their", "its", "his",
    "her", "that", "this", "these", "those", "do", "does", "did", "have", "has",
    "had", "as", "it", "they", "them", "there",
}


def question_terms(text):
    words = re.findall(r"[a-z0-9]+|n't", text.lower())
    return [w for w in words if w not in STOPWORDS]


pool_terms = [question_terms(get_question(r)) for r in train_pool]
document_frequency = collections.Counter(t for terms in pool_terms for t in set(terms))
pool_size = len(train_pool)


def tfidf(terms):
    counts = collections.Counter(terms)
    vector = {t: (1 + math.log(c)) * math.log((1 + pool_size) / (1 + document_frequency.get(t, 0)))
              for t, c in counts.items()}
    norm = math.sqrt(sum(v * v for v in vector.values())) or 1.0
    return {t: v / norm for t, v in vector.items()}


pool_vectors = [tfidf(terms) for terms in pool_terms]


def eligible(shot, example):
    """Never show the test question itself or one of its paraphrases."""
    if shot.get("id") is not None and shot.get("id") == example.get("id"):
        return False
    if args.exclude_same_db and shot.get("db_id") == example.get("db_id"):
        return False
    group = example.get("paraphrase_group")
    return not (group and shot.get("paraphrase_group") == group)


def rank_shots(example):
    """Eligible training examples, most similar first ('random': pool order)."""
    candidates = [i for i, shot in enumerate(train_pool) if eligible(shot, example)]
    if args.shot_strategy == "random":
        return candidates
    query = tfidf(question_terms(get_question(example)))
    scored = []
    for i in candidates:
        vector = pool_vectors[i]
        score = sum(weight * vector.get(term, 0.0) for term, weight in query.items())
        if train_pool[i]["db_id"] == example["db_id"]:
            score += args.same_db_bonus
        scored.append((score, i))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [i for _, i in scored]


def shots_for_iteration(example, index, iteration):
    """
    The shots of one iteration, most similar first.

    Shot sets are drawn without replacement of whole sets: iteration k never
    reuses a set from iterations 0..k-1 of the same question, unless the
    pool has fewer combinations than iterations. The random generator is
    seeded with (seed, question id), and the sets of iterations 0..k are
    drawn in order, so the result does not depend on how iterations are
    split across jobs.
    """
    ranked = rank_shots(example)
    pool = ranked if args.shot_strategy == "random" else ranked[:args.shot_pool_size]
    k = min(args.num_shots, len(pool))
    combinations = math.comb(len(pool), k)
    rng = random.Random(f"{args.seed}:{example.get('id', index)}")
    used = set()
    chosen = None
    for _ in range(iteration + 1):
        for _attempt in range(1000):
            pick = tuple(sorted(rng.sample(range(len(pool)), k)))
            if pick not in used or len(used) >= combinations:
                break
        used.add(pick)
        chosen = pick
    return [train_pool[pool[i]] for i in chosen]


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=os.environ.get("HF_TOKEN"))
vocab = tokenizer.get_vocab()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

eos_ids = [vocab[t] for t in STOP_TOKENS if t in vocab]
if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in eos_ids:
    eos_ids.append(tokenizer.eos_token_id)

print(f"\nLoading {args.model_name} in 4-bit NF4...")
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ),
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    token=os.environ.get("HF_TOKEN"),
)
model.eval()
device = next(model.parameters()).device

generation_kwargs = {
    "max_new_tokens": args.max_new_tokens,
    "do_sample": args.do_sample,
    "num_beams": 1,
    "repetition_penalty": 1.0,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": eos_ids,
    "use_cache": True,
}
if args.do_sample:
    generation_kwargs.update(temperature=args.temperature, top_p=args.top_p)
else:
    generation_kwargs.update(temperature=None, top_p=None, top_k=None)


def gemma_messages(messages):
    """Gemma 2's chat template has no system role (it raises an error), so
    the rules are put at the top of the first user turn."""
    if messages and messages[0]["role"] == "system":
        system, rest = messages[0]["content"], [dict(m) for m in messages[1:]]
        rest[0]["content"] = system + "\n\n" + rest[0]["content"]
        return rest
    return messages


def render(messages):
    return tokenizer.apply_chat_template(gemma_messages(messages), tokenize=False,
                                         add_generation_prompt=True)


def build_prompt(example, index, iteration):
    """Prompt with this iteration's demonstrations, fitted to the token
    budget. The most similar shot is placed last, next to the question."""
    ranked = shots_for_iteration(example, index, iteration)
    while True:
        shots = list(reversed(ranked))
        prompt = render(build_messages(example, schema_map, few_shot=shots))
        length = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if length <= args.max_input_tokens or not ranked:
            return prompt, shots, length
        ranked = ranked[:-1]          # drop the least similar shot


def generate(prompts):
    encoded = tokenizer(prompts, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(device)
    with torch.inference_mode():
        output = model.generate(**encoded, **generation_kwargs)
    new_tokens = output[:, encoded["input_ids"].shape[1]:]
    return [tokenizer.decode(t, skip_special_tokens=True) for t in new_tokens]


# ------------------------------------------------------------
# EVALUATION HELPERS
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


def exact_match_key(sql):
    return normalize_sql(sql).replace('"', "'").lower()


def finish_record(example, shots, prompt_tokens, raw_output, iteration, error=None):
    predicted_sql = extract_sql(raw_output) if raw_output else ""
    gold_sql = get_sql(example)
    record = dict(example)
    record.update(
        predicted_sql=predicted_sql,
        raw_output=raw_output,
        exact_match=bool(predicted_sql) and exact_match_key(predicted_sql) == exact_match_key(gold_sql),
        shot_ids=[s.get("id") for s in shots],
        num_shots_used=len(shots),
        prompt_tokens=prompt_tokens,
        model=args.model_name,
        mode=args.mode,
        num_shots=args.num_shots,
        shot_strategy=args.shot_strategy,
        shot_pool_size=args.shot_pool_size,
        seed=args.seed,
        iteration=iteration,
    )
    if error:
        record["error"] = error
    if args.database_dir:
        if predicted_sql:
            gold_answer = example.get("answer")
            if gold_answer is None:
                gold_answer, _ = execute(example["db_id"], gold_sql)
            pred_rows, pred_error = execute(example["db_id"], predicted_sql)
            ordered = bool(re.search(r"\border\s+by\b", gold_sql, re.I))
            record["execution_error"] = pred_error
            record["execution_match"] = results_match(gold_answer, pred_rows, ordered)
            partner = paired.get(example.get("id"))
            if partner is not None:
                record["matches_paired_answer"] = (
                    results_match(partner.get("answer"), pred_rows, ordered)
                    and not record["execution_match"]
                )
        else:
            record["execution_match"] = False
            record["execution_error"] = error or "empty_prediction"
    return record


# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------

def percent(rows, field):
    return round(100.0 * sum(bool(r.get(field)) for r in rows) / len(rows), 2) if rows else None


def iteration_path(iteration):
    base = Path(args.output_file)
    return base.with_name(f"{base.stem}_iter{iteration}{base.suffix or '.jsonl'}")


def run_iteration(iteration):
    output_path = iteration_path(iteration)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = set()
    if args.resume and output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    done_ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    todo = [(i, row) for i, row in enumerate(test_data) if row.get("id") not in done_ids]

    print("\n" + "=" * 80)
    print(f"ITERATION {iteration}   already done: {len(done_ids)}   to run: {len(todo)}")
    print(f"Output: {output_path}")
    print("=" * 80)

    too_long = 0
    with output_path.open("a" if args.resume else "w", encoding="utf-8") as out:

        def flush(batch):
            try:
                outputs = generate([item[3] for item in batch])
                errors = [None] * len(batch)
            except Exception as exc:  # e.g. out of memory: retry one by one
                torch.cuda.empty_cache()
                outputs, errors = [], []
                for item in batch:
                    try:
                        outputs.append(generate([item[3]])[0])
                        errors.append(None)
                    except Exception as single_exc:
                        outputs.append("")
                        errors.append(str(single_exc) or str(exc))
            for (example, shots, length, _), raw, error in zip(batch, outputs, errors):
                record = finish_record(example, shots, length, raw, iteration, error)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                mark = ""
                if "execution_match" in record:
                    mark = "OK " if record["execution_match"] else "BAD"
                print(f"[it{iteration} {example.get('id')}] {mark} shots={len(shots)} "
                      f"{record['predicted_sql']}")
            out.flush()

        batch = []
        for index, example in todo:
            prompt, shots, length = build_prompt(example, index, iteration)
            if length > args.max_input_tokens:
                too_long += 1
                record = finish_record(example, [], length, "", iteration,
                                       error="prompt_longer_than_max_input_tokens")
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue
            batch.append((example, shots, length, prompt))
            if len(batch) == args.batch_size:
                flush(batch)
                batch = []
        if batch:
            flush(batch)

    with output_path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    summary = {
        "model": args.model_name,
        "mode": args.mode,
        "iteration": iteration,
        "test_file": args.test_file,
        "train_file": args.train_file,
        "num_shots": args.num_shots,
        "shot_strategy": args.shot_strategy,
        "shot_pool_size": args.shot_pool_size,
        "same_db_bonus": args.same_db_bonus,
        "exclude_same_db": args.exclude_same_db,
        "seed": args.seed,
        "examples": len(records),
        "empty_predictions": sum(1 for r in records if not r.get("predicted_sql")),
        "prompts_too_long": too_long,
        "examples_with_fewer_shots": sum(1 for r in records
                                         if r.get("num_shots_used", 0) < args.num_shots),
        "exact_match": percent(records, "exact_match"),
    }
    if args.database_dir:
        summary["execution_accuracy"] = percent(records, "execution_match")
        summary["execution_errors"] = sum(1 for r in records if r.get("execution_error"))
        if paired:
            summary["returns_paired_answer_percent"] = percent(records, "matches_paired_answer")
        for field in ("negation_type", "question_rule", "source_dataset"):
            groups = collections.defaultdict(list)
            for record in records:
                if record.get(field):
                    groups[record[field]].append(record)
            if groups:
                summary[f"accuracy_by_{field}"] = {
                    key: {"n": len(rows), "accuracy": percent(rows, "execution_match")}
                    for key, rows in sorted(groups.items())
                }

    with output_path.with_suffix(".summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\nIteration {iteration}: exact match {summary['exact_match']}  "
          f"execution accuracy {summary.get('execution_accuracy')}")
    return summary, {r["id"]: tuple(r.get("shot_ids", [])) for r in records if "id" in r}


iterations = range(args.start_iteration, args.start_iteration + args.num_iterations)
summaries = {}
shot_sets = {}
for iteration in iterations:
    summaries[iteration], shot_sets[iteration] = run_iteration(iteration)


# ------------------------------------------------------------
# ACROSS ITERATIONS
# ------------------------------------------------------------

def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    mean = sum(values) / len(values)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    return {"mean": round(mean, 2), "std": round(std, 2), "values": values}


repeated = 0
seen = collections.defaultdict(set)
for iteration in iterations:
    for example_id, shots in shot_sets[iteration].items():
        if shots and shots in seen[example_id]:
            repeated += 1
        seen[example_id].add(shots)

overall = {
    "iterations": list(iterations),
    "repeated_shot_sets": repeated,
    "per_iteration": {str(i): {k: summaries[i].get(k) for k in
                               ("execution_accuracy", "exact_match", "returns_paired_answer_percent",
                                "execution_errors", "examples")}
                      for i in iterations},
}
for metric in ("execution_accuracy", "exact_match", "returns_paired_answer_percent"):
    overall[metric] = mean_std([summaries[i].get(metric) for i in iterations])

overall_path = Path(args.output_file).with_suffix(".iterations_summary.json")
if len(iterations) > 1:
    with overall_path.open("w", encoding="utf-8") as handle:
        json.dump(overall, handle, indent=2)

print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)
for iteration in iterations:
    s_ = summaries[iteration]
    print(f"iteration {iteration}: exec acc {s_.get('execution_accuracy')}  "
          f"exact match {s_.get('exact_match')}  paired {s_.get('returns_paired_answer_percent')}")
for metric in ("execution_accuracy", "exact_match", "returns_paired_answer_percent"):
    if overall[metric]:
        print(f"{metric:<30}: {overall[metric]['mean']} +/- {overall[metric]['std']}")
print(f"Shot sets repeated across iterations: {repeated}")
for iteration in iterations:
    print(f"Predictions it{iteration}: {iteration_path(iteration)}")
if len(iterations) > 1:
    print(f"Across iterations : {overall_path}")
print("=" * 80)