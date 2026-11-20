#!/usr/bin/env python3
"""Inference/evaluation for Qwen2.5-7B-Instruct CWQ QA adapters or few-shot prompts."""
import argparse, ast, json, random, re
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM = ("Answer the question using your knowledge. Return only a valid JSON array of answer "
         "strings. Include each answer as a separate string. Do not include explanations or markdown.")

def load_records(path):
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text: raise ValueError(f"Empty input file: {path}")
    try: obj = json.loads(text)
    except json.JSONDecodeError:
        obj = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(obj, dict):
        for k in ("data", "examples", "records", "questions"):
            if isinstance(obj.get(k), list): obj = obj[k]; break
        else: obj = [obj]
    if not isinstance(obj, list) or not all(isinstance(x, dict) for x in obj):
        raise ValueError(f"Expected a list of JSON objects: {path}")
    return obj

def answers(value):
    if value is None: return []
    if isinstance(value, list): vals = value
    elif isinstance(value, str):
        value = value.strip()
        if not value: return []
        try:
            parsed = ast.literal_eval(value)
            vals = parsed if isinstance(parsed, (list, tuple, set)) else [parsed]
        except (ValueError, SyntaxError): vals = [value]
    else: vals = [value]
    return [str(x).strip() for x in vals if x is not None and str(x).strip()]

def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())

def parse_output(text):
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.I)
    left, right = text.find("["), text.rfind("]")
    if left >= 0 and right > left:
        candidate = text[left:right+1]
        for parser in (json.loads, ast.literal_eval):
            try:
                value = parser(candidate)
                if isinstance(value, (list, tuple)): return answers(list(value)), True
            except (ValueError, SyntaxError, json.JSONDecodeError): pass
    try:
        value = json.loads(text)
        if isinstance(value, str) and value.strip(): return [value.strip()], True
    except json.JSONDecodeError: pass
    cleaned = text.strip().strip("'").strip('"')
    return ([cleaned] if cleaned else []), False

def metrics(rows):
    exact = hit = tp = fp = fn = 0
    for r in rows:
        p = {norm(x) for x in r["predicted_answers"] if norm(x)}
        g = {norm(x) for x in r["gold_answers"] if norm(x)}
        overlap = p & g
        exact += int(p == g)
        hit += int(bool(overlap))
        tp += len(overlap); fp += len(p-overlap); fn += len(g-overlap)
    n = len(rows)
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    return {
        "num_examples": n,
        "exact_match": exact/n if n else 0.0,
        "hit_at_1": hit/n if n else 0.0,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": 2*precision*recall/(precision+recall) if precision+recall else 0.0,
        "true_positive_answers": tp, "false_positive_answers": fp, "false_negative_answers": fn
    }

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test_file", required=True)
    ap.add_argument("--output_file", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter_dir", default=None, help="LoRA adapter directory; omit for base/few-shot inference")
    ap.add_argument("--fewshot_file", default=None, help="Training QA file for few-shot examples")
    ap.add_argument("--num_shots", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--max_input_length", type=int, default=4096)
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="Evaluate first N examples; 0 means all")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.num_shots < 0 or args.limit < 0: ap.error("--num_shots and --limit must be non-negative")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU not detected.")
    test_path = Path(args.test_file)
    if not test_path.is_file(): ap.error(f"Test file not found: {test_path}")
    test_rows = load_records(test_path)
    if args.limit: test_rows = test_rows[:args.limit]

    shots = []
    if args.fewshot_file:
        sp = Path(args.fewshot_file)
        if not sp.is_file(): ap.error(f"Few-shot file not found: {sp}")
        candidates = [r for r in load_records(sp) if isinstance(r.get("question"), str)
                      and r.get("question", "").strip() and answers(r.get("answers"))]
        random.Random(args.seed).shuffle(candidates)
        shots = candidates[:args.num_shots]
        print(f"Selected {len(shots)} few-shot examples from training file.")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype) if args.load_in_4bit else None
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"torch_dtype": dtype, "device_map": "auto"}
    if quant is not None: kwargs["quantization_config"] = quant
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **kwargs)
    if args.adapter_dir:
        if not Path(args.adapter_dir).is_dir(): ap.error(f"Adapter directory not found: {args.adapter_dir}")
        model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()
    input_device = model.get_input_embeddings().weight.device
    results = []
    for idx, row in enumerate(test_rows, 1):
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            print(f"[{idx}/{len(test_rows)}] skipped: missing question")
            continue
        gold = answers(row.get("answers"))
        messages = [{"role": "system", "content": SYSTEM}]
        for shot in shots:
            messages += [
                {"role": "user", "content": str(shot["question"]).strip()},
                {"role": "assistant", "content": json.dumps(answers(shot["answers"]), ensure_ascii=False)}
            ]
        messages.append({"role": "user", "content": question.strip()})
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_input_length)
        batch = {k: v.to(input_device) for k, v in batch.items()}
        with torch.inference_mode():
            generated = model.generate(**batch, max_new_tokens=args.max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        output = tokenizer.decode(generated[0][batch["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        pred, parsed = parse_output(output)
        results.append({"id": row.get("id", str(idx-1)), "question": question,
            "gold_answers": gold, "predicted_answers": pred, "raw_generation": output,
            "answer_parse_success": parsed})
        if idx == 1 or idx % 25 == 0 or idx == len(test_rows):
            print(f"[{idx}/{len(test_rows)}] parsed={parsed}; prediction={pred[:3]}")

    score = metrics(results)
    payload = {"model_name": args.model_name, "adapter_dir": args.adapter_dir,
        "fewshot_file": args.fewshot_file, "num_shots": len(shots),
        "test_file": str(test_path), "metrics": score, "predictions": results}
    out = Path(args.output_file); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\nMetrics")
    for k, v in score.items(): print(f"{k}: {v}")
    print(f"\nSaved predictions and metrics to: {out.resolve()}")

if __name__ == "__main__":
    main()
