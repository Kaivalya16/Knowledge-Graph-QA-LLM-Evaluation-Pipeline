#!/usr/bin/env python3
"""QLoRA fine-tuning for Qwen2.5-7B-Instruct on CWQ QA data."""
import argparse
import ast
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import SFTConfig, SFTTrainer


def load_records(path):
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty input file: {path}")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = []
        for n, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    obj.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON/JSONL at line {n}: {exc}") from exc
    if isinstance(obj, dict):
        for key in ("data", "examples", "records", "questions"):
            if isinstance(obj.get(key), list):
                obj = obj[key]
                break
        else:
            obj = [obj]
    if not isinstance(obj, list) or not all(isinstance(x, dict) for x in obj):
        raise ValueError("Expected a JSON array of objects or JSONL records.")
    return obj


def answer_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        vals = value
    elif isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = ast.literal_eval(value)
            vals = parsed if isinstance(parsed, (list, tuple)) else [parsed]
        except (ValueError, SyntaxError):
            vals = [value]
    else:
        vals = [value]
    return [str(x).strip() for x in vals if x is not None and str(x).strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_file", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--num_train_epochs", type=float, default=3)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--no_4bit", action="store_true",
                   help="Disable 4-bit quantization; requires substantially more GPU memory.")
    args = p.parse_args()

    train_path = Path(args.train_file)
    if not train_path.is_file():
        p.error(f"Training file not found: {train_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected; this script is configured for GPU fine-tuning.")
    if args.max_seq_length < 64:
        p.error("--max_seq_length should be at least 64")

    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    rows = load_records(train_path)
    examples = []
    skipped = 0
    for row in rows:
        question = row.get("question")
        answers = answer_list(row.get("answers"))
        if not isinstance(question, str) or not question.strip() or not answers:
            skipped += 1
            continue
        messages = [
            {"role": "system", "content":
             "Answer the question using your knowledge. Return only a valid JSON array "
             "of answer strings, with no explanation or markdown."},
            {"role": "user", "content": question.strip()},
            {"role": "assistant", "content": json.dumps(answers, ensure_ascii=False)},
        ]
        examples.append({
            "text": tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        })
    if not examples:
        raise ValueError("No usable examples: check 'question' and 'answers' fields.")
    print(f"Input records: {len(rows)} | usable: {len(examples)} | skipped: {skipped}")
    print("Training example preview:\n", examples[0]["text"][:1200])

    quant_config = None if args.no_4bit else BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    model_kwargs = {"torch_dtype": dtype, "device_map": "auto"}
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.config.use_cache = False

    if quant_config is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=2,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        optim="paged_adamw_8bit" if quant_config is not None else "adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_length=args.max_seq_length,
        packing=False,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none",
        seed=args.seed,
        dataset_text_field="text",
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(examples),
        peft_config=lora,
        processing_class=tokenizer,
    )
    print("Starting fine-tuning...")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Done. LoRA adapter and tokenizer saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
