#!/usr/bin/env python3
"""
Gemma-2-9B-IT QLoRA fine-tuning for Spider positive / negative Text-to-SQL.

Same pipeline, prompt content (text2sql_prompt.py), data handling and
options as 01_finetune_qwen.py / 01_finetune_llama.py. Gemma-specific parts:

- google/gemma-2-9b-it is a gated model: accept the licence on its Hugging
  Face page and log in once (`huggingface-cli login`).
- Gemma 2's chat template has no system role, so the rules are placed at the
  top of the first user turn (the same text the other models get as a system
  message).
- The model turn is closed by <end_of_turn>; the training sequence is
  prompt tokens + SQL + <end_of_turn> and is checked against the template.
  The newline the template adds after <end_of_turn> is not trained.
- Attention is "eager": Gemma 2 uses attention-logit soft-capping, which the
  other attention kernels do not apply during training.
- Gemma has its own <pad> token, so no padding workaround is needed.
- LoRA r=16 / alpha=32 on q,k,v,o,gate,up,down projections, loss on the SQL
  and <end_of_turn> only.
- Memory: Gemma's vocabulary has 256,000 tokens, and the standard loss
  builds a float32 logits matrix for EVERY position (about 1 GB per 1,000
  tokens, plus its gradient). The long baseball_1 schemas (~2,400 tokens)
  therefore ran out of memory on a 20 GB GPU. SqlOnlyLossTrainer computes
  the logits only at the SQL positions that carry the loss (a few dozen),
  which gives the same loss and gradients with far less memory.
  --full_logits_loss restores the standard loss. The embedding/output
  matrix is also kept in bfloat16 (peft's prepare_model_for_kbit_training
  would up-cast it to float32: +1.8 GB).

Example:
  python 01_finetune_gemma.py --mode pos \\
      --input_file final_datasets/pos_train.json \\
      --tables_file data/raw/tables.json --output_dir adapters/gemma_pos
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import inspect
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text2sql_prompt import build_messages, get_sql, load_schema_map, normalize_sql  # noqa: E402


MODEL_NAME = "google/gemma-2-9b-it"

END_OF_TURN = "<end_of_turn>"


# ------------------------------------------------------------
# ARGUMENTS
# ------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["pos", "neg", "mix"], required=True,
                        help="Label for the run; 'mix' expects several input files.")
    parser.add_argument("--input_file", nargs="+", required=True,
                        help="One or more training JSON files.")
    parser.add_argument("--tables_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default=MODEL_NAME)

    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="If > 0, overrides --warmup_ratio.")
    parser.add_argument("--max_length", type=int, default=4096)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--val_ratio", type=float, default=0.0,
                        help="Hold out this share of the TRAIN data (by paraphrase "
                             "group) for eval loss. 0 = no validation.")
    parser.add_argument("--no_schema_keys", action="store_true",
                        help="Schema without types/keys (v3 format).")

    parser.add_argument("--full_logits_loss", action="store_true",
                        help="Standard Trainer loss (logits at every position; "
                             "much more memory).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Use only the first N examples.")
    parser.add_argument("--debug_overfit", action="store_true",
                        help="20 samples, 50 epochs, LR 2e-4, grad accumulation 1.")

    args = parser.parse_args()
    if args.mode != "mix" and len(args.input_file) > 1:
        parser.error("Several --input_file values need --mode mix.")
    return args


args = parse_args()

if args.debug_overfit:
    args.max_train_samples = 20
    args.num_train_epochs = 50
    args.learning_rate = 2e-4
    args.gradient_accumulation_steps = 1
    args.val_ratio = 0.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(args.seed)

print("=" * 80)
print("Gemma-2-9B-IT QLoRA Fine-tuning")
print("=" * 80)
for key, value in vars(args).items():
    print(f"{key:<28}: {value}")
print("=" * 80)


# ------------------------------------------------------------
# TOKENIZER AND SCHEMA
# ------------------------------------------------------------

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading Spider tables...")
schema_map = load_schema_map(
    args.tables_file,
    with_types=not args.no_schema_keys,
    with_keys=not args.no_schema_keys,
)
print(f"Loaded {len(schema_map)} databases.")


def gemma_messages(messages):
    """Gemma 2's chat template has no system role (it raises an error), so
    the rules are put at the top of the first user turn."""
    if messages and messages[0]["role"] == "system":
        system, rest = messages[0]["content"], [dict(m) for m in messages[1:]]
        rest[0]["content"] = system + "\n\n" + rest[0]["content"]
        return rest
    return messages


def prompt_text(example):
    return tokenizer.apply_chat_template(
        gemma_messages(build_messages(example, schema_map, with_answer=False)),
        tokenize=False, add_generation_prompt=True,
    )


def full_text(example):
    return tokenizer.apply_chat_template(
        gemma_messages(build_messages(example, schema_map, with_answer=True)),
        tokenize=False, add_generation_prompt=False,
    )


def answer_text(example):
    return normalize_sql(get_sql(example)) + END_OF_TURN


# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

print("\nLoading training JSON...")
raw_data = []
for path in args.input_file:
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    for row in rows:
        row = dict(row)
        row["_source_file"] = os.path.basename(path)
        raw_data.append(row)
    print(f"  {path}: {len(rows)} examples")

if args.max_train_samples is not None:
    raw_data = raw_data[:args.max_train_samples]
print(f"Input examples: {len(raw_data)}")


def split_validation(rows, ratio, seed):
    """Hold out whole paraphrase groups so no question leaks into training."""
    if ratio <= 0:
        return rows, []
    groups = sorted({r.get("paraphrase_group") or r.get("id") for r in rows})
    random.Random(seed).shuffle(groups)
    held = set(groups[:max(1, round(ratio * len(groups)))])
    train = [r for r in rows if (r.get("paraphrase_group") or r.get("id")) not in held]
    val = [r for r in rows if (r.get("paraphrase_group") or r.get("id")) in held]
    return train, val


train_rows, val_rows = split_validation(raw_data, args.val_ratio, args.seed)
print(f"Train examples: {len(train_rows)}   validation examples: {len(val_rows)}")


# ------------------------------------------------------------
# TOKENIZATION
# ------------------------------------------------------------

def tokenize_rows(rows, label):
    tokenized, lengths, prompt_lengths, sql_lengths = [], [], [], []
    skipped_long, skipped_bad = [], []

    for index, example in enumerate(rows):
        try:
            prompt = prompt_text(example)
            answer = answer_text(example)
            # The chat template must render prompt + SQL + <end_of_turn> + "\n";
            # the final newline is not trained.
            if full_text(example).rstrip("\n") != prompt + answer:
                raise ValueError("Chat template does not end the turn with SQL + <end_of_turn>.")

            # The template text already starts with <bos>, so no special
            # tokens are added (that would give a second BOS).
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
            full_ids = prompt_ids + answer_ids

            prompt_len = len(prompt_ids)
            sql_len = len(answer_ids)
            if sql_len <= 0:
                raise ValueError("No assistant SQL tokens detected.")

            lengths.append(len(full_ids))
            prompt_lengths.append(prompt_len)
            sql_lengths.append(sql_len)

            # Never keep only part of an oversized schema: skip instead.
            if len(full_ids) > args.max_length:
                skipped_long.append({"index": index, "id": example.get("id"),
                                     "total_tokens": len(full_ids),
                                     "prompt_tokens": prompt_len, "sql_tokens": sql_len})
                continue

            tokenized.append({
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": [-100] * prompt_len + full_ids[prompt_len:],
            })
        except Exception as exc:
            skipped_bad.append({"index": index, "id": example.get("id"), "error": str(exc)})

    print("\n" + "=" * 80)
    print(f"{label.upper()} DATA DIAGNOSTICS")
    print("=" * 80)
    print(f"Examples seen           : {len(rows)}")
    print(f"Examples used           : {len(tokenized)}")
    print(f"Skipped > max_length    : {len(skipped_long)}")
    print(f"Skipped due to errors   : {len(skipped_bad)}")
    if skipped_bad:
        print(f"  first error           : {skipped_bad[0]['error']}")
    if lengths:
        print(f"Total tokens min/avg/max: {min(lengths)} / {np.mean(lengths):.1f} / {max(lengths)}")
        print(f"Average prompt tokens   : {np.mean(prompt_lengths):.1f}")
        print(f"Average SQL tokens      : {np.mean(sql_lengths):.1f}")
        for limit in (512, 1024, 2048, 4096):
            count = sum(x > limit for x in lengths)
            print(f"Examples > {limit:4d} tokens: {count}/{len(lengths)} "
                  f"({100.0 * count / len(lengths):.2f}%)")
    print("=" * 80)
    return tokenized, skipped_long, skipped_bad


train_tokenized, skipped_long, skipped_bad = tokenize_rows(train_rows, "training")
if not train_tokenized:
    raise RuntimeError("No usable training examples remain - see the errors above.")
val_tokenized = tokenize_rows(val_rows, "validation")[0] if val_rows else []

sample = train_rows[0]
print("\nTRAINING SAMPLE")
print("=" * 80)
print(full_text(sample))
print(f"Target SQL: {normalize_sql(get_sql(sample))}")
print("=" * 80)

steps_per_epoch = max(1, len(train_tokenized) //
                      (args.per_device_train_batch_size * args.gradient_accumulation_steps))
total_steps = int(steps_per_epoch * args.num_train_epochs)
warmup = args.warmup_steps if args.warmup_steps > 0 else int(args.warmup_ratio * total_steps)
print(f"\nOptimizer steps: ~{total_steps} ({steps_per_epoch}/epoch), warmup steps: {warmup}")

train_dataset = Dataset.from_list(train_tokenized)
val_dataset = Dataset.from_list(val_tokenized) if val_tokenized else None


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

print("\nLoading model in 4-bit NF4...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    trust_remote_code=True,
)
model.config.use_cache = False

# prepare_model_for_kbit_training would also up-cast every non-quantized
# weight to float32. For Gemma that includes the 256k x 3584 embedding /
# output matrix (+1.8 GB), so only its gradient-checkpointing part is done
# here and those weights stay in bfloat16.
for parameter in model.parameters():
    parameter.requires_grad = False
try:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
except TypeError:  # older transformers
    model.gradient_checkpointing_enable()
model.enable_input_require_grads()

lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)
print("\nLoRA trainable parameters:")
model.print_trainable_parameters()


# ------------------------------------------------------------
# TRAINING ARGUMENTS (version tolerant)
# ------------------------------------------------------------

def make_training_arguments(**kwargs):
    supported = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in kwargs and "eval_strategy" not in supported:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    dropped = [k for k in kwargs if k not in supported]
    for key in dropped:
        kwargs.pop(key)
    if dropped:
        print(f"Note: this Transformers version ignores {dropped}")
    return TrainingArguments(**kwargs)


training_args = make_training_arguments(
    output_dir=args.output_dir,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    lr_scheduler_type="cosine",
    warmup_steps=warmup,
    weight_decay=0.01,
    logging_strategy="steps",
    logging_steps=1 if args.debug_overfit else 5,
    eval_strategy="epoch" if val_dataset is not None else "no",
    save_strategy="epoch",
    save_total_limit=2,
    bf16=True,
    fp16=False,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    remove_unused_columns=False,
    prediction_loss_only=True,
    report_to="none",
    seed=args.seed,
    data_seed=args.seed,
    dataloader_num_workers=0,
    ddp_find_unused_parameters=False,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,
    pad_to_multiple_of=8,
)

class SqlOnlyLossTrainer(Trainer):
    """Cross-entropy on the SQL tokens only, computing logits only there.

    Same loss as Gemma2ForCausalLM (including its final logit soft-capping),
    but the 256k-wide logits are built for the labelled positions instead of
    the whole sequence. Each micro-batch returns its mean token loss; the
    Trainer averages over gradient-accumulation steps.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        # always let the Trainer divide by gradient_accumulation_steps
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        causal_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
        hidden = causal_lm.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
        )[0]
        target = labels[:, 1:]
        mask = target != -100
        selected = hidden[:, :-1, :][mask]
        logits = causal_lm.lm_head(selected).float()
        cap = getattr(causal_lm.config, "final_logit_softcapping", None)
        if cap:
            logits = torch.tanh(logits / cap) * cap
        loss = torch.nn.functional.cross_entropy(logits, target[mask])
        return (loss, {"loss": loss}) if return_outputs else loss


TrainerClass = Trainer if args.full_logits_loss else SqlOnlyLossTrainer
print(f"\nLoss: {'standard (all logits)' if args.full_logits_loss else 'SQL positions only'}")

trainer = TrainerClass(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)


# ------------------------------------------------------------
# TRAIN AND SAVE
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("STARTING TRAINING")
print("=" * 80)
train_result = trainer.train()
print(f"\nTraining loss: {train_result.training_loss:.4f}")

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
trainer.save_model(str(output_dir))
tokenizer.save_pretrained(str(output_dir))

config = dict(vars(args), model_name=args.model_name, warmup_steps_used=warmup,
              optimizer_steps=total_steps, train_examples=len(train_tokenized),
              val_examples=len(val_tokenized))
files = {
    "training_config.json": config,
    "loss_history.json": trainer.state.log_history,
    "skipped_long_examples.json": skipped_long,
    "skipped_bad_examples.json": skipped_bad,
    "validation_ids.json": [r.get("id") for r in val_rows],
}
for name, content in files.items():
    with open(output_dir / name, "w", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2)

print(f"\nAdapter saved to: {output_dir}")
print("Done.")