#!/usr/bin/env python3
"""Gemma-2-9B-IT QLoRA fine-tuning for CWQ QA datasets."""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse, ast, json, random
from pathlib import Path
import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq, Trainer, TrainingArguments

DEFAULT_MODEL = "google/gemma-2-9b-it"
SYSTEM_PROMPT = """You are a question-answering system for ComplexWebQuestions (CWQ).
Answer the question using your learned knowledge. Return only a valid JSON array of answer strings.
Examples: ["Paris"] or ["Alice", "Bob"]; if there is no answer, return [].
Do not include explanations, markdown, or text outside the JSON array."""

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Gemma-2-9B-IT on CWQ using QLoRA.")
    p.add_argument("--mode", choices=["pos", "neg"], required=True)
    p.add_argument("--input_file", required=True, help="Prepared CWQ QA JSON with id, question, answers.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_name", default=DEFAULT_MODEL)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--debug_overfit", action="store_true")
    return p.parse_args()

args = parse_args()
if args.debug_overfit:
    args.max_train_samples, args.num_train_epochs = 20, 50
    args.learning_rate, args.gradient_accumulation_steps = 2e-4, 1

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
print("=" * 72)
print("Gemma-2-9B-IT QLoRA Fine-tuning for CWQ")
print(f"Mode: {args.mode} | Model: {args.model_name}")
print(f"Input: {args.input_file}")
print(f"Output: {args.output_dir}")
print(f"Epochs={args.num_train_epochs}, LR={args.learning_rate}, max_length={args.max_length}")
print("=" * 72)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Check nvidia-smi and your CUDA-enabled PyTorch installation.")

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def parse_answers(value):
    if value is None: return []
    if isinstance(value, (list, tuple)): values = list(value)
    elif isinstance(value, str):
        value = value.strip()
        if not value: return []
        try: parsed = json.loads(value)
        except json.JSONDecodeError:
            try: parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError): parsed = [value]
        values = list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
    else: values = [value]
    return [str(x).strip() for x in values if x is not None and str(x).strip()]

def question_of(ex):
    q = ex.get("question")
    if not isinstance(q, str) or not q.strip():
        raise ValueError(f"Missing/empty question; keys={list(ex.keys())}")
    return q.strip()

def answer_of(ex):
    if "answers" not in ex: raise KeyError(f"Missing answers; keys={list(ex.keys())}")
    return json.dumps(parse_answers(ex["answers"]), ensure_ascii=False)

def user_content(ex): return f"{SYSTEM_PROMPT}\n\nQuestion:\n{question_of(ex)}"
def prompt_text(ex):
    return tokenizer.apply_chat_template([{"role":"user","content":user_content(ex)}], tokenize=False, add_generation_prompt=True)
def full_text(ex):
    return tokenizer.apply_chat_template([{"role":"user","content":user_content(ex)}, {"role":"assistant","content":answer_of(ex)}], tokenize=False, add_generation_prompt=False)

with open(args.input_file, encoding="utf-8") as f: raw = json.load(f)
if isinstance(raw, dict):
    for key in ("data", "examples", "records"):
        if isinstance(raw.get(key), list): raw = raw[key]; break
if not isinstance(raw, list): raise ValueError("Input must be a JSON array of examples.")
if args.max_train_samples is not None: raw = raw[:args.max_train_samples]
if not raw: raise RuntimeError("No training examples found.")

items=[]; lengths=[]; prompt_lens=[]; answer_lens=[]; skipped_long=[]; skipped_bad=[]
for i, ex in enumerate(raw):
    try:
        if not isinstance(ex, dict): raise TypeError("Each example must be a JSON object.")
        pids = tokenizer(prompt_text(ex), add_special_tokens=False)["input_ids"]
        ids = tokenizer(full_text(ex), add_special_tokens=False)["input_ids"]
        plen, total = len(pids), len(ids)
        alen = total - plen
        lengths.append(total); prompt_lens.append(plen); answer_lens.append(alen)
        if alen <= 0: raise ValueError("No assistant answer tokens detected.")
        if total > args.max_length:
            skipped_long.append({"index":i,"id":ex.get("id"),"total_tokens":total,"prompt_tokens":plen,"answer_tokens":alen})
            continue
        items.append({"input_ids":ids,"attention_mask":[1]*total,"labels":[-100]*plen+ids[plen:]})
    except Exception as e:
        skipped_bad.append({"index":i,"id":ex.get("id") if isinstance(ex,dict) else None,"error":str(e)})
if not items: raise RuntimeError("No usable examples. Check fields and --max_length.")
print(f"Examples: {len(raw)} | used: {len(items)} | too long: {len(skipped_long)} | errors: {len(skipped_bad)}")
print(f"Average tokens: {np.mean(lengths):.1f}; average prompt: {np.mean(prompt_lens):.1f}; average answer: {np.mean(answer_lens):.1f}")
print("Sample question:", question_of(raw[0]))
print("Sample target:", answer_of(raw[0]))

print("Loading model in 4-bit NF4...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(args.model_name, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                  bias="none", task_type="CAUSAL_LM",
                  target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
model = get_peft_model(model, lora)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=args.output_dir, num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=1, gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate, lr_scheduler_type="cosine", warmup_steps=args.warmup_steps,
    weight_decay=0.01, logging_strategy="steps", logging_steps=1 if args.debug_overfit else 10,
    save_strategy="steps", save_steps=100 if args.debug_overfit else 500, save_total_limit=2,
    bf16=True, fp16=False, optim="paged_adamw_8bit", gradient_checkpointing=True,
    remove_unused_columns=False, report_to="none", seed=args.seed, data_seed=args.seed,
    dataloader_num_workers=0, ddp_find_unused_parameters=False)
collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True,
                                  label_pad_token_id=-100, pad_to_multiple_of=8)
trainer = Trainer(model=model, args=training_args, train_dataset=Dataset.from_list(items), data_collator=collator)
print("Starting training...")
result = trainer.train()
print("Training loss:", result.training_loss)
out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
trainer.save_model(str(out)); tokenizer.save_pretrained(str(out))
with open(out/"training_config.json","w",encoding="utf-8") as f: json.dump(vars(args),f,indent=2)
with open(out/"skipped_long_examples.json","w",encoding="utf-8") as f: json.dump(skipped_long,f,indent=2,ensure_ascii=False)
with open(out/"skipped_bad_examples.json","w",encoding="utf-8") as f: json.dump(skipped_bad,f,indent=2,ensure_ascii=False)
print("LoRA adapter saved to:", out)
