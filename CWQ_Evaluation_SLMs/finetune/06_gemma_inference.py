#!/usr/bin/env python3
"""Inference for Gemma-2-9B-IT on CWQ, with LoRA or few-shot prompting."""
import argparse
import ast
import json
import random
import re
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_MODEL = "google/gemma-2-9b-it"
SYSTEM_PROMPT = """You are a question-answering system for ComplexWebQuestions (CWQ).
Answer the question using your learned knowledge.
Return only a valid JSON array of answer strings.
Examples: ["Paris"] or ["Alice", "Bob"]; if there is no answer, return [].
Do not include explanations, markdown, or text outside the JSON array."""


def parse_args():
    p = argparse.ArgumentParser(description="Gemma-2-9B-IT CWQ inference")
    p.add_argument("--test_file", required=True, help="CWQ QA test JSON.")
    p.add_argument("--output_file", required=True, help="Output predictions JSON.")
    p.add_argument("--adapter_dir", default=None, help="Fine-tuned LoRA adapter. Omit for few-shot.")
    p.add_argument("--fewshot_file", default=None, help="Training JSON used as few-shot examples.")
    p.add_argument("--num_shots", type=int, default=4)
    p.add_argument("--model_name", default=DEFAULT_MODEL)
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--max_input_length", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def load_json_list(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("data", "examples", "records"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array of examples.")
    return data


def parse_answers(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        vals = list(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                parsed = [text]
        vals = list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
    else:
        vals = [value]
    return [str(x).strip() for x in vals if x is not None and str(x).strip()]


def extract_answers(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parse_answers(parsed), True
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[[\s\S]*?\]", text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return parse_answers(parsed), True
        except json.JSONDecodeError:
            pass
    return ([text] if text else []), False


def build_messages(question, shots):
    messages = []
    for ex in shots:
        q = str(ex.get("question", "")).strip()
        if not q:
            continue
        target = json.dumps(parse_answers(ex.get("answers", [])), ensure_ascii=False)
        messages.append({"role": "user", "content": f"{SYSTEM_PROMPT}\n\nQuestion:\n{q}"})
        messages.append({"role": "assistant", "content": target})
    messages.append({"role": "user", "content": f"{SYSTEM_PROMPT}\n\nQuestion:\n{question}"})
    return messages


def normalize_answer(s):
    s = str(s).strip().lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def score_answers(predicted, gold):
    pred = [normalize_answer(x) for x in predicted]
    gold = [normalize_answer(x) for x in gold]
    pred = [x for x in pred if x]
    gold = [x for x in gold if x]
    remaining = list(gold)
    matched = 0
    for answer in pred:
        for i, target in enumerate(remaining):
            if answer == target:
                matched += 1
                remaining.pop(i)
                break
    precision = matched / len(pred) if pred else (1.0 if not gold else 0.0)
    recall = matched / len(gold) if gold else (1.0 if not pred else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "exact_set_match": set(pred) == set(gold),
        "hit_at_1": matched > 0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "num_predicted": len(predicted),
        "num_gold": len(gold),
        "num_matched": matched,
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if bool(args.adapter_dir) == bool(args.fewshot_file):
        raise ValueError("Specify exactly one: --adapter_dir OR --fewshot_file.")
    if args.num_shots < 0:
        raise ValueError("--num_shots must be >= 0.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Check nvidia-smi and CUDA-enabled PyTorch.")

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    print(f"Loading base model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    if args.adapter_dir:
        print(f"Loading adapter: {args.adapter_dir}")
        model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

    test_data = load_json_list(args.test_file)
    if args.limit is not None:
        test_data = test_data[:args.limit]

    shots = []
    if args.fewshot_file:
        train_data = load_json_list(args.fewshot_file)
        if args.num_shots:
            shots = random.sample(train_data, min(args.num_shots, len(train_data)))
        print(f"Few-shot examples selected: {len(shots)}")

    predictions = []
    for idx, ex in enumerate(tqdm(test_data, desc="CWQ inference")):
        question = str(ex.get("question", "")).strip()
        gold = parse_answers(ex.get("answers", []))
        if not question:
            predictions.append({
                "id": ex.get("id", str(idx)), "question": question,
                "gold_answers": gold, "predicted_answers": [],
                "raw_generation": "", "answer_parse_success": False,
                "error": "Missing/empty question",
            })
            continue

        prompt = tokenizer.apply_chat_template(
            build_messages(question, shots), tokenize=False, add_generation_prompt=True
        )
        encoded = tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=args.max_input_length,
        )
        input_device = model.get_input_embeddings().weight.device
        encoded = {k: v.to(input_device) for k, v in encoded.items()}
        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if args.temperature > 0:
            gen_kwargs["temperature"] = args.temperature
            gen_kwargs["top_p"] = args.top_p
        with torch.no_grad():
            output_ids = model.generate(**encoded, **gen_kwargs)
        new_ids = output_ids[0][encoded["input_ids"].shape[1]:]
        raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        predicted, parsed_ok = extract_answers(raw)
        scores = score_answers(predicted, gold)
        predictions.append({
            "id": ex.get("id", str(idx)),
            "question": question,
            "gold_answers": gold,
            "predicted_answers": predicted,
            "raw_generation": raw,
            "answer_parse_success": parsed_ok,
            **scores,
        })

    valid = [x for x in predictions if "error" not in x]
    n = len(valid)
    summary = {
        "model_name": args.model_name,
        "adapter_dir": args.adapter_dir,
        "fewshot_file": args.fewshot_file,
        "num_shots": len(shots),
        "test_file": args.test_file,
        "num_examples": len(test_data),
        "num_evaluated": n,
        "parse_success_rate": sum(x["answer_parse_success"] for x in valid) / n if n else 0.0,
        "exact_set_match": sum(x["exact_set_match"] for x in valid) / n if n else 0.0,
        "hit_at_1": sum(x["hit_at_1"] for x in valid) / n if n else 0.0,
        "precision": sum(x["precision"] for x in valid) / n if n else 0.0,
        "recall": sum(x["recall"] for x in valid) / n if n else 0.0,
        "f1": sum(x["f1"] for x in valid) / n if n else 0.0,
    }
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "predictions": predictions}, f, indent=2, ensure_ascii=False)
    print("\n" + "=" * 70)
    print("CWQ INFERENCE COMPLETE")
    print("=" * 70)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Saved predictions to: {out_path}")


if __name__ == "__main__":
    main()
