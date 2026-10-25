#!/usr/bin/env python3
"""
Gemma-2-9B-IT QLoRA fine-tuning for Spider Text-to-SQL.

v3 changes:
- 4096-token context by default.
- SQL/assistant-only loss.
- Never truncates schema/prompt from the left.
- Long examples are skipped and reported.
- LoRA r=16 / alpha=32 on attention + MLP projections.
- Full training defaults: 3 epochs, LR 1e-4, grad accumulation 16.
- --debug_overfit automatically switches to a real memorization test:
    * max 20 examples
    * 50 epochs
    * LR 2e-4
    * gradient accumulation 1
- Compatible with Transformers versions that do not support group_by_length.
- Saves adapter, tokenizer, configuration, and skipped-example report.
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


MODEL_NAME = "google/gemma-2-9b-it"

SYSTEM_PROMPT = """You are an expert Text-to-SQL system.

Convert the natural language question into a valid SQLite SQL query.

Rules:
1. Use only the tables and columns present in the provided schema.
2. Do not invent tables or columns.
3. Use a JOIN only when the question requires information from multiple tables
   or when the schema requires a relationship between tables.
4. Preserve all filtering, comparison, aggregation, ordering, grouping,
   join, and negation requirements.
5. Return only the SQL query.
6. Do not provide explanations.
7. Do not use markdown code fences.
"""


def parse_args():
    parser = argparse.ArgumentParser()
 
    parser.add_argument("--mode", choices=["pos", "neg"], required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--tables_file", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--warmup_steps", type=float, default=100)
    parser.add_argument("--max_length", type=int, default=4096)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Use only the first N examples."
    )

    parser.add_argument(
        "--debug_overfit",
        action="store_true",
        help=(
            "Run the controlled 20-example memorization test: "
            "20 samples, 50 epochs, LR 2e-4, gradient accumulation 1."
        ),
    )

    return parser.parse_args()


args = parse_args()

# ------------------------------------------------------------
# DEBUG OVERFIT MODE
# ------------------------------------------------------------
if args.debug_overfit:
    args.max_train_samples = 20
    args.num_train_epochs = 50
    args.learning_rate = 2e-4
    args.gradient_accumulation_steps = 1

# ------------------------------------------------------------
# REPRODUCIBILITY
# ------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(args.seed)

print("=" * 80)
print("Gemma-2-9B-IT QLoRA Fine-tuning v3")
print("=" * 80)
print(f"Mode                    : {args.mode}")
print(f"Model                   : {MODEL_NAME}")
print(f"Input                   : {args.input_file}")
print(f"Tables                  : {args.tables_file}")
print(f"Output                  : {args.output_dir}")
print(f"Epochs                  : {args.num_train_epochs}")
print(f"Learning rate           : {args.learning_rate}")
print(f"Grad accumulation       : {args.gradient_accumulation_steps}")
print(f"Warmup steps            : {args.warmup_steps}")
print(f"Max length              : {args.max_length}")
print(f"LoRA r                  : {args.lora_r}")
print(f"LoRA alpha              : {args.lora_alpha}")
print(f"LoRA dropout            : {args.lora_dropout}")
print(f"Max samples             : {args.max_train_samples}")
print(f"Debug overfit           : {args.debug_overfit}")
print(f"Seed                    : {args.seed}")
print("=" * 80)

# ------------------------------------------------------------
# TOKENIZER
# ------------------------------------------------------------
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

# ------------------------------------------------------------
# SCHEMA
# ------------------------------------------------------------
print("Loading Spider tables...")

with open(args.tables_file, "r", encoding="utf-8") as f:
    tables_data = json.load(f)


def build_schema_map(tables):
    result = {}

    for db in tables:
        db_id = db["db_id"]
        table_names = db["table_names_original"]
        columns = db["column_names_original"]

        table_columns = {name: [] for name in table_names}

        for table_idx, column_name in columns:
            if table_idx == -1:
                continue

            table_name = table_names[table_idx]
            table_columns[table_name].append(column_name)

        lines = []
        for table_name in table_names:
            lines.append(
                f"Table {table_name}: "
                f"{', '.join(table_columns[table_name])}"
            )

        result[db_id] = "\n".join(lines)

    return result


schema_map = build_schema_map(tables_data)
print(f"Loaded {len(schema_map)} databases.")

# ------------------------------------------------------------
# DATA FIELD HELPERS
# ------------------------------------------------------------
def get_question(example):
    if args.mode == "pos":
        if "question" in example:
            return example["question"]
    else:
        for key in ("negated_question", "negative_question"):
            if key in example:
                return example[key]

    raise KeyError(
        f"Question field not found for mode={args.mode}. "
        f"Available keys: {list(example.keys())}"
    )


def get_sql(example):
    if args.mode == "pos":
        for key in ("sql", "query"):
            if key in example:
                return example[key]
    else:
        for key in ("negated_sql", "negative_sql", "sql", "query"):
            if key in example:
                return example[key]

    raise KeyError(
        f"SQL field not found for mode={args.mode}. "
        f"Available keys: {list(example.keys())}"
    )


def clean_sql(sql):
    sql = str(sql).strip()

    if sql.startswith("```sql"):
        sql = sql[len("```sql"):]
    elif sql.startswith("```SQL"):
        sql = sql[len("```SQL"):]

    sql = sql.replace("```", "")
    return sql.strip()


# ------------------------------------------------------------
# PROMPT
# ------------------------------------------------------------
def build_user_content(example):
    db_id = example["db_id"]

    if db_id not in schema_map:
        raise KeyError(
            f"Database '{db_id}' is not present in tables file."
        )

    return f"""{SYSTEM_PROMPT}

Database:
{db_id}

Schema:
{schema_map[db_id]}

Question:
{get_question(example)}"""


def build_prompt_text(example):
    messages = [
        {
            "role": "user",
            "content": build_user_content(example),
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_full_text(example):
    messages = [
        {
            "role": "user",
            "content": build_user_content(example),
        },
        {
            "role": "assistant",
            "content": clean_sql(get_sql(example)),
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
print("\nLoading training JSON...")

with open(args.input_file, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

if args.max_train_samples is not None:
    raw_data = raw_data[:args.max_train_samples]

print(f"Input examples: {len(raw_data)}")

# ------------------------------------------------------------
# TOKENIZATION
# ------------------------------------------------------------
tokenized = []
all_lengths = []
prompt_lengths = []
sql_lengths = []
skipped_long = []
skipped_bad = []

print("\nTokenizing examples...")
print("Loss is applied ONLY to assistant SQL tokens.")

for idx, example in enumerate(raw_data):
    try:
        prompt_text = build_prompt_text(example)
        full_text = build_full_text(example)

        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]

        full_ids = tokenizer(
            full_text,
            add_special_tokens=False,
        )["input_ids"]

        prompt_len = len(prompt_ids)
        full_len = len(full_ids)
        sql_len = full_len - prompt_len

        all_lengths.append(full_len)
        prompt_lengths.append(prompt_len)
        sql_lengths.append(sql_len)

        if sql_len <= 0:
            raise ValueError("No assistant SQL tokens detected.")

        # IMPORTANT:
        # Never keep only the end of an oversized schema.
        # Skip the example instead.
        if full_len > args.max_length:
            skipped_long.append(
                {
                    "index": idx,
                    "id": example.get("id"),
                    "total_tokens": full_len,
                    "prompt_tokens": prompt_len,
                    "sql_tokens": sql_len,
                }
            )
            continue

        labels = [-100] * prompt_len + full_ids[prompt_len:]

        tokenized.append(
            {
                "input_ids": full_ids,
                "attention_mask": [1] * full_len,
                "labels": labels,
            }
        )

    except Exception as exc:
        skipped_bad.append(
            {
                "index": idx,
                "id": example.get("id"),
                "error": str(exc),
            }
        )


if not tokenized:
    raise RuntimeError(
        "No usable training examples remain."
    )

# ------------------------------------------------------------
# DIAGNOSTICS
# ------------------------------------------------------------
print("\n" + "=" * 80)
print("TRAINING DATA DIAGNOSTICS")
print("=" * 80)

total = len(all_lengths)
used = len(tokenized)

print(f"Examples seen              : {total}")
print(f"Examples used              : {used}")
print(f"Skipped > max_length      : {len(skipped_long)}")
print(f"Skipped due to errors      : {len(skipped_bad)}")

if total:
    print(f"Minimum total tokens       : {min(all_lengths)}")
    print(f"Maximum total tokens       : {max(all_lengths)}")
    print(f"Average total tokens       : {np.mean(all_lengths):.1f}")
    print(f"Average prompt tokens      : {np.mean(prompt_lengths):.1f}")
    print(f"Average SQL tokens         : {np.mean(sql_lengths):.1f}")

    for limit in (512, 1024, 2048, 4096):
        count = sum(x > limit for x in all_lengths)
        pct = 100.0 * count / total
        print(
            f"Examples > {limit:4d} tokens  : "
            f"{count}/{total} ({pct:.2f}%)"
        )

print("=" * 80)

# ------------------------------------------------------------
# SAMPLE CHECK
# ------------------------------------------------------------
sample = raw_data[0]

sample_prompt = build_prompt_text(sample)
sample_full = build_full_text(sample)

sample_prompt_ids = tokenizer(
    sample_prompt,
    add_special_tokens=False,
)["input_ids"]

sample_full_ids = tokenizer(
    sample_full,
    add_special_tokens=False,
)["input_ids"]

print("\n" + "=" * 80)
print("TRAINING SAMPLE CHECK")
print("=" * 80)
print(f"ID             : {sample.get('id')}")
print(f"Database       : {sample['db_id']}")
print(f"Prompt tokens  : {len(sample_prompt_ids)}")
print(
    f"SQL tokens     : "
    f"{len(sample_full_ids) - len(sample_prompt_ids)}"
)
print(f"Total tokens   : {len(sample_full_ids)}")
print("\nTarget SQL:")
print(clean_sql(get_sql(sample)))
print("=" * 80)

if args.debug_overfit:
    print("\nDEBUG OVERFIT CONFIGURATION")
    print("20 examples")
    print("50 epochs")
    print("learning rate = 2e-4")
    print("gradient accumulation = 1")
    print("This is intentionally designed to memorize the examples.")

# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------
train_dataset = Dataset.from_list(tokenized)

# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------
print("\nLoading Gemma-2-9B-IT in 4-bit NF4...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

model.config.use_cache = False

# ------------------------------------------------------------
# QLORA
# ------------------------------------------------------------
model = prepare_model_for_kbit_training(model)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()

lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model = get_peft_model(model, lora_config)

print("\nLoRA trainable parameters:")
model.print_trainable_parameters()

# ------------------------------------------------------------
# TRAINING ARGUMENTS
# ------------------------------------------------------------
# NOTE:
# Do NOT use group_by_length here. Your installed Transformers
# version does not support that argument.

training_args = TrainingArguments(
    output_dir=args.output_dir,

    num_train_epochs=args.num_train_epochs,

    per_device_train_batch_size=1,
    gradient_accumulation_steps=args.gradient_accumulation_steps,

    learning_rate=args.learning_rate,
    lr_scheduler_type="cosine",
    warmup_steps=args.warmup_steps,

    weight_decay=0.01,

    logging_strategy="steps",
    logging_steps=1 if args.debug_overfit else 10,

    save_strategy="steps",
    save_steps=100 if args.debug_overfit else 500,
    save_total_limit=2,

    bf16=True,
    fp16=False,

    optim="paged_adamw_8bit",

    gradient_checkpointing=True,

    remove_unused_columns=False,

    report_to="none",

    seed=args.seed,
    data_seed=args.seed,

    dataloader_num_workers=0,

    ddp_find_unused_parameters=False,
)

# ------------------------------------------------------------
# COLLATOR
# ------------------------------------------------------------
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,
    pad_to_multiple_of=8,
)

# ------------------------------------------------------------
# TRAINER
# ------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------
print("\n" + "=" * 80)
print("STARTING TRAINING")
print("=" * 80)

train_result = trainer.train()

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"Training loss: {train_result.training_loss}")

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

trainer.save_model(str(output_dir))
tokenizer.save_pretrained(str(output_dir))

with open(output_dir / "training_config.json", "w", encoding="utf-8") as f:
    json.dump(vars(args), f, indent=2)

with open(output_dir / "skipped_long_examples.json", "w", encoding="utf-8") as f:
    json.dump(skipped_long, f, indent=2)

with open(output_dir / "skipped_bad_examples.json", "w", encoding="utf-8") as f:
    json.dump(skipped_bad, f, indent=2)

print(f"\nAdapter saved to: {output_dir}")
print("Done.")
