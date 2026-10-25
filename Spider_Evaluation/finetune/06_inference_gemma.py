#!/usr/bin/env python3
"""
Gemma-2-9B-IT LoRA inference for Spider Text-to-SQL.

Uses the same prompt format as 05_finetune_gemma_v3.py.
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
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
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--output_file", required=True)

    parser.add_argument("--max_input_length", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=256)

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Run inference on only the first N examples."
    )

    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Use sampling. Default is deterministic generation."
    )

    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)

    return parser.parse_args()


args = parse_args()

print("=" * 80)
print("Gemma-2-9B-IT LoRA Inference v3")
print("=" * 80)
print(f"Mode             : {args.mode}")
print(f"Input             : {args.input_file}")
print(f"Tables            : {args.tables_file}")
print(f"Adapter           : {args.adapter_dir}")
print(f"Output            : {args.output_file}")
print(f"Max input tokens  : {args.max_input_length}")
print(f"Max new tokens    : {args.max_new_tokens}")
print(f"Max samples       : {args.max_samples}")
print(f"Sampling          : {args.do_sample}")
print("=" * 80)

# ------------------------------------------------------------
# TOKENIZER
# ------------------------------------------------------------
print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"

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
# QUESTION
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
        f"Question field not found. Keys: {list(example.keys())}"
    )


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


def build_prompt(example):
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


# ------------------------------------------------------------
# SQL CLEANING
# ------------------------------------------------------------
def clean_sql(text):
    text = text.strip()

    # Markdown fences.
    text = re.sub(
        r"```(?:sql)?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = text.replace("```", "").strip()

    # Common prefixes.
    for prefix in ("SQL:", "sql:", "Answer:", "answer:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    # Keep only the first SQL statement.
    if ";" in text:
        text = text.split(";", 1)[0]

    return text.strip()


# ------------------------------------------------------------
# INPUT
# ------------------------------------------------------------
print("\nLoading input JSON...")

with open(args.input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

if args.max_samples is not None:
    data = data[:args.max_samples]

print(f"Examples to process: {len(data)}")

# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------
print("\nLoading base Gemma-2-9B-IT in 4-bit NF4...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

print("Base model loaded.")

print("\nLoading LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    args.adapter_dir,
)

model.eval()

device = next(model.parameters()).device
print(f"Inference device: {device}")

# ------------------------------------------------------------
# GENERATION
# ------------------------------------------------------------
def generate_sql(example):
    prompt = build_prompt(example)

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_input_length,
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    prompt_length = encoded["input_ids"].shape[1]

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if args.do_sample:
        generation_kwargs.update(
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
        )

    with torch.inference_mode():
        outputs = model.generate(
            **encoded,
            **generation_kwargs,
        )

    generated_ids = outputs[0, prompt_length:]

    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return clean_sql(generated_text)


# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------
output_path = Path(args.output_file)
output_path.parent.mkdir(parents=True, exist_ok=True)

successful = 0
failed = 0

print("\n" + "=" * 80)
print("STARTING INFERENCE")
print("=" * 80)

with output_path.open("w", encoding="utf-8", buffering=1) as out:
    for i, example in enumerate(data):
        try:
            predicted_sql = generate_sql(example)

            result = dict(example)
            result["predicted_sql"] = predicted_sql
            result["model"] = MODEL_NAME
            result["adapter_dir"] = args.adapter_dir
            result["mode"] = args.mode

            out.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )
            out.flush()

            successful += 1

            print(
                f"[{i + 1}/{len(data)}] "
                f"OK | {predicted_sql}"
            )

        except Exception as exc:
            failed += 1

            result = dict(example)
            result["predicted_sql"] = ""
            result["model"] = MODEL_NAME
            result["adapter_dir"] = args.adapter_dir
            result["mode"] = args.mode
            result["error"] = str(exc)

            out.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )
            out.flush()

            print(
                f"[{i + 1}/{len(data)}] ERROR: {exc}"
            )

print("\n" + "=" * 80)
print("INFERENCE COMPLETE")
print("=" * 80)
print(f"Total     : {len(data)}")
print(f"Successful: {successful}")
print(f"Failed    : {failed}")
print(f"Output    : {args.output_file}")
print("=" * 80)
