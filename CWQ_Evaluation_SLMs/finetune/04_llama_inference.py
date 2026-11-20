#!/usr/bin/env python3
"""Inference/evaluation for Llama-3.1-8B-Instruct CWQ QA adapters or few-shot prompts."""
import argparse, ast, json, random, re
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM = ("Answer the question using your knowledge. Return only a valid JSON array of answer "
         "strings. Include each answer separately. Do not include explanations or markdown.")

def load_records(path):
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text: raise ValueError(f"Empty input file: {path}")
    try: obj = json.loads(text)
    except json.JSONDecodeError: obj = [json.loads(x) for x in text.splitlines() if x.strip()]
    if isinstance(obj, dict):
        for key in ("data", "examples", "records", "questions"):
            if isinstance(obj.get(key), list): obj = obj[key]; break
        else: obj = [obj]
    if not isinstance(obj, list) or not all(isinstance(x, dict) for x in obj):
        raise ValueError(f"Expected JSON list of objects: {path}")
    return obj

def answers(value):
    if value is None: return []
    if isinstance(value, list): vals = value
    elif isinstance(value, str):
        value = value.strip()
        if not value: return []
        try:
            v = ast.literal_eval(value)
            vals = v if isinstance(v, (list, tuple, set)) else [v]
        except (ValueError, SyntaxError): vals = [value]
    else: vals = [value]
    return [str(x).strip() for x in vals if x is not None and str(x).strip()]

def norm(text):
    text = re.sub(r"\b(a|an|the)\b", " ", str(text).lower().strip())
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())

def parse_output(text):
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.I)
    start, end = text.find("["), text.rfind("]")
    if start >= 0 and end > start:
        candidate = text[start:end+1]
        for fn in (json.loads, ast.literal_eval):
            try:
                val = fn(candidate)
                if isinstance(val, (list, tuple)): return answers(list(val)), True
            except (ValueError, SyntaxError, json.JSONDecodeError): pass
    try:
        val = json.loads(text)
        if isinstance(val, str) and val.strip(): return [val.strip()], True
    except json.JSONDecodeError: pass
    text = text.strip().strip("'").strip('"')
    return ([text] if text else []), False

def score(rows):
    em = hit = tp = fp = fn = 0
    for r in rows:
        p = {norm(x) for x in r["predicted_answers"] if norm(x)}
        g = {norm(x) for x in r["gold_answers"] if norm(x)}
        common = p & g
        em += int(p == g); hit += int(bool(common))
        tp += len(common); fp += len(p-common); fn += len(g-common)
    n = len(rows); prec = tp/(tp+fp) if tp+fp else 0.0
    rec = tp/(tp+fn) if tp+fn else 0.0
    return {"num_examples": n, "exact_match": em/n if n else 0.0,
            "hit_at_1": hit/n if n else 0.0, "micro_precision": prec,
            "micro_recall": rec, "micro_f1": 2*prec*rec/(prec+rec) if prec+rec else 0.0,
            "true_positive_answers": tp, "false_positive_answers": fp, "false_negative_answers": fn}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test_file", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--adapter_dir", default=None)
    p.add_argument("--fewshot_file", default=None)
    p.add_argument("--num_shots", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--max_input_length", type=int, default=4096)
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if a.num_shots < 0 or a.limit < 0: p.error("--num_shots and --limit must be non-negative")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU not detected.")
    test_path = Path(a.test_file)
    if not test_path.is_file(): p.error(f"Test file not found: {test_path}")
    test = load_records(test_path)
    if a.limit: test = test[:a.limit]
    shots = []
    if a.fewshot_file:
        sf = Path(a.fewshot_file)
        if not sf.is_file(): p.error(f"Few-shot file not found: {sf}")
        candidates = [r for r in load_records(sf) if isinstance(r.get("question"), str)
                      and r["question"].strip() and answers(r.get("answers"))]
        random.Random(a.seed).shuffle(candidates)
        shots = candidates[:a.num_shots]
        print(f"Selected {len(shots)} few-shot examples from training data.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype) if a.load_in_4bit else None
    print(f"Loading tokenizer: {a.model_name}")
    tok = AutoTokenizer.from_pretrained(a.model_name, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    kwargs = {"torch_dtype": dtype, "device_map": "auto"}
    if quant is not None: kwargs["quantization_config"] = quant
    model = AutoModelForCausalLM.from_pretrained(a.model_name, **kwargs)
    if a.adapter_dir:
        if not Path(a.adapter_dir).is_dir(): p.error(f"Adapter directory not found: {a.adapter_dir}")
        model = PeftModel.from_pretrained(model, a.adapter_dir)
    model.eval()
    device = model.get_input_embeddings().weight.device
    predictions = []
    for i, row in enumerate(test, 1):
        q = row.get("question")
        if not isinstance(q, str) or not q.strip():
            print(f"[{i}/{len(test)}] skipped missing question"); continue
        msgs = [{"role": "system", "content": SYSTEM}]
        for ex in shots:
            msgs += [{"role": "user", "content": ex["question"].strip()},
                     {"role": "assistant", "content": json.dumps(answers(ex["answers"]), ensure_ascii=False)}]
        msgs.append({"role": "user", "content": q.strip()})
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=a.max_input_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        raw = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        pred, parsed = parse_output(raw)
        predictions.append({"id": row.get("id", str(i-1)), "question": q,
            "gold_answers": answers(row.get("answers")), "predicted_answers": pred,
            "raw_generation": raw, "answer_parse_success": parsed})
        if i == 1 or i % 25 == 0 or i == len(test):
            print(f"[{i}/{len(test)}] parsed={parsed}; prediction={pred[:3]}")
    metrics = score(predictions)
    result = {"model_name": a.model_name, "adapter_dir": a.adapter_dir,
        "fewshot_file": a.fewshot_file, "num_shots": len(shots),
        "test_file": str(test_path), "metrics": metrics, "predictions": predictions}
    out = Path(a.output_file); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print("\nEvaluation metrics")
    for k, v in metrics.items(): print(f"{k}: {v}")
    print(f"Saved results to: {out.resolve()}")

if __name__ == "__main__":
    main()
