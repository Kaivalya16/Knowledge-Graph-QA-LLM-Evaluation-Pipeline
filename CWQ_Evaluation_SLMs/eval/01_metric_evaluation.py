#!/usr/bin/env python3
"""
CWQ answer evaluation, adapted from the user's Spider metric script.

Supports:
  1) Prediction JSON produced by inference_qwen25_7b_cwq.py /
     inference_llama31_8b_cwq.py (top-level "predictions" array)
  2) A JSON array of prediction records
  3) JSONL prediction records

Gold file should be the matching CWQ QA test file, e.g. cwq_pos_test.json or
cwq_neg_test.json, with records containing id, question, and answers.

CWQ adaptation:
  - Answers are strings, not SQL result rows.
  - EM matches normalized answer strings exactly.
  - PEM matches if one normalized answer string is a prefix of the other.
  - Answer matching is one-to-one and order-independent.
  - Dataset averages use evaluated questions as the denominator, matching
    the structure of the supplied Spider evaluator.
"""

import argparse
import ast
import json
import re
from pathlib import Path


# ============================================================
# LOAD JSON / JSONL
# ============================================================

def load_json_or_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        raise ValueError(f"Input file is empty: {path}")

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = []
        for line_num, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"Warning: invalid JSON on line {line_num}: {exc}")

    # Inference scripts save a wrapper object containing "predictions".
    if isinstance(data, dict):
        if isinstance(data.get("predictions"), list):
            data = data["predictions"]
        else:
            for key in ("data", "examples", "records", "questions"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
            else:
                data = [data]

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array/object or JSONL file: {path}")
    return data


# ============================================================
# NORMALIZE ANSWERS
# ============================================================

def to_answer_list(value):
    """Convert arrays or list-like strings into a list of answer strings."""
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        values = list(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            values = list(parsed) if isinstance(parsed, (list, tuple, set)) else [parsed]
        except (ValueError, SyntaxError):
            values = [text]
    else:
        values = [value]

    return [
        str(item).strip()
        for item in values
        if item is not None and str(item).strip()
    ]


def normalize_answer(value):
    """
    Normalize a CWQ answer for matching.

    Case-insensitive; removes punctuation and common English articles while
    preserving letters, numbers, and word order.
    """
    text = str(value).lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def normalize_answers(values):
    return [normalize_answer(value) for value in to_answer_list(values)]


# ============================================================
# EXACT / PARTIAL ANSWER MATCH
# ============================================================

def partial_answer_match(predicted_answer, gold_answer):
    """
    CWQ equivalent of Spider's partial row match:
    one normalized answer must be a prefix of the other.
    """
    if predicted_answer == gold_answer:
        return True
    if not predicted_answer or not gold_answer:
        return False
    return (
        gold_answer.startswith(predicted_answer)
        or predicted_answer.startswith(gold_answer)
    )


def match_answers(predicted, gold, partial=False):
    """Greedy one-to-one matching; order of answers does not matter."""
    matched_gold = set()
    per_prediction = []

    for pred_answer in predicted:
        found = False
        for gold_idx, gold_answer in enumerate(gold):
            if gold_idx in matched_gold:
                continue

            matches = (
                partial_answer_match(pred_answer, gold_answer)
                if partial else pred_answer == gold_answer
            )
            if matches:
                matched_gold.add(gold_idx)
                found = True
                break

        per_prediction.append(int(found))

    return per_prediction, matched_gold


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def evaluate_prediction(gold_answer, predicted_answer):
    gold = normalize_answers(gold_answer)
    pred = normalize_answers(predicted_answer)

    em_scores, matched_gold_em = match_answers(pred, gold, partial=False)
    pem_scores, matched_gold_pem = match_answers(pred, gold, partial=True)

    hit_em = int(any(em_scores))
    hit_pem = int(any(pem_scores))

    num_pred = len(pred)
    num_gold = len(gold)

    precision_em = safe_ratio(sum(em_scores), num_pred)
    recall_em = safe_ratio(len(matched_gold_em), num_gold)
    f1_em = (
        2 * precision_em * recall_em / (precision_em + recall_em)
        if precision_em + recall_em else 0.0
    )

    tp_em = len(matched_gold_em)
    fp_em = num_pred - tp_em
    fn_em = num_gold - tp_em
    accuracy_em = safe_ratio(tp_em, tp_em + fp_em + fn_em)

    precision_pem = safe_ratio(sum(pem_scores), num_pred)
    recall_pem = safe_ratio(len(matched_gold_pem), num_gold)
    f1_pem = (
        2 * precision_pem * recall_pem / (precision_pem + recall_pem)
        if precision_pem + recall_pem else 0.0
    )

    tp_pem = len(matched_gold_pem)
    fp_pem = num_pred - tp_pem
    fn_pem = num_gold - tp_pem
    accuracy_pem = safe_ratio(tp_pem, tp_pem + fp_pem + fn_pem)

    # Per-question exact-set equality, unlike the Spider script's EM_avg.
    exact_set_match = int(
        sorted(set(x for x in pred if x)) == sorted(set(x for x in gold if x))
    )

    return {
        "EM_scores_per_answer": em_scores,
        "PEM_scores_per_answer": pem_scores,
        "HIT_EM": hit_em,
        "HIT_PEM": hit_pem,
        "Exact_Set_Match": exact_set_match,
        "EM_avg": safe_ratio(sum(em_scores), num_pred),
        "PEM_avg": safe_ratio(sum(pem_scores), num_pred),
        "Precision_EM": precision_em,
        "Recall_EM": recall_em,
        "F1_EM": f1_em,
        "Accuracy_EM": accuracy_em,
        "Precision_PEM": precision_pem,
        "Recall_PEM": recall_pem,
        "F1_PEM": f1_pem,
        "Accuracy_PEM": accuracy_pem,
        "num_predicted_answers": num_pred,
        "num_gold_answers": num_gold,
        "matched_exact_answers": tp_em,
        "matched_partial_answers": tp_pem,
    }


# ============================================================
# MATCH PREDICTIONS TO GOLD
# ============================================================

def get_gold_answer(example):
    for key in ("gold_answers", "gold_answer", "answers", "answer"):
        if key in example:
            return example[key]
    raise KeyError(f"Gold answer field not found. Available fields: {list(example.keys())}")


def build_gold_lookup(gold_data):
    lookup = {}
    for idx, example in enumerate(gold_data):
        if example.get("id") is not None:
            lookup[("id", str(example["id"]))] = example
        if isinstance(example.get("question"), str):
            lookup[("question", example["question"])] = example
        lookup[("position", idx)] = example
    return lookup


def find_gold_example(prediction, gold_lookup, positional_idx):
    # 1. Stable record ID
    if prediction.get("id") is not None:
        key = ("id", str(prediction["id"]))
        if key in gold_lookup:
            return gold_lookup[key]

    # 2. Exact question text
    if isinstance(prediction.get("question"), str):
        key = ("question", prediction["question"])
        if key in gold_lookup:
            return gold_lookup[key]

    # 3. Positional fallback only if the prediction record has no usable ID/question.
    # Avoid accidentally pairing a mismatched question with an unrelated gold record.
    if prediction.get("id") is None and not prediction.get("question"):
        return gold_lookup.get(("position", positional_idx))
    return None


# ============================================================
# MAIN EVALUATION
# ============================================================

def run_evaluation(predictions_file, gold_file, output_file):
    print("Loading predictions...")
    predictions = load_json_or_jsonl(predictions_file)
    print("Loading gold CWQ QA test data...")
    gold_data = load_json_or_jsonl(gold_file)

    print(f"Predictions available: {len(predictions)}")
    print(f"Gold test examples:     {len(gold_data)}")

    gold_lookup = build_gold_lookup(gold_data)
    evaluated = []
    unmatched = 0

    for idx, prediction in enumerate(predictions):
        if not isinstance(prediction, dict):
            unmatched += 1
            continue

        gold_example = find_gold_example(prediction, gold_lookup, idx)
        if gold_example is None:
            unmatched += 1
            continue

        gold_answer = get_gold_answer(gold_example)
        predicted_answer = prediction.get(
            "predicted_answers",
            prediction.get("predicted_answer", prediction.get("prediction", []))
        )

        metric_values = evaluate_prediction(gold_answer, predicted_answer)
        record = dict(prediction)
        record["gold_answers"] = to_answer_list(gold_answer)
        record["predicted_answers"] = to_answer_list(predicted_answer)
        record.update(metric_values)
        evaluated.append(record)

        if (idx + 1) % 100 == 0 or idx + 1 == len(predictions):
            percent = (idx + 1) / max(len(predictions), 1) * 100
            print(f"\rEvaluating: {percent:6.2f}% ({idx + 1}/{len(predictions)})", end="", flush=True)

    print("\n")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(evaluated, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )

    total_q = len(evaluated)
    if total_q == 0:
        print("No matching predictions found. Check IDs/questions in the prediction and gold files.")
        return

    def mean(key):
        return sum(float(row.get(key, 0.0)) for row in evaluated) / total_q

    total_hit_em = sum(row["HIT_EM"] for row in evaluated)
    total_hit_pem = sum(row["HIT_PEM"] for row in evaluated)
    total_set_em = sum(row["Exact_Set_Match"] for row in evaluated)

    print("=" * 64)
    print("                 CWQ ANSWER EVALUATION")
    print("=" * 64)
    print(f"Gold test questions:       {len(gold_data)}")
    print(f"Predictions provided:      {len(predictions)}")
    print(f"Predictions evaluated:     {total_q}")
    print(f"Predictions unmatched:     {unmatched}")
    print("\nOnly predictions matched to the supplied gold file are evaluated.")

    print("\n--- Hit Rates ---")
    print(f"Total EM HITs:              {total_hit_em}")
    print(f"Average EM HIT Rate:        {mean('HIT_EM'):.4f}")
    print(f"Total PEM HITs:             {total_hit_pem}")
    print(f"Average PEM HIT Rate:       {mean('HIT_PEM'):.4f}")

    print("\n--- Exact Match (EM) ---")
    print(f"Mean EM Avg:                {mean('EM_avg'):.4f}")
    print(f"Mean Precision:             {mean('Precision_EM'):.4f}")
    print(f"Mean Recall:                {mean('Recall_EM'):.4f}")
    print(f"Mean F1:                    {mean('F1_EM'):.4f}")
    print(f"Mean Accuracy:              {mean('Accuracy_EM'):.4f}")
    print(f"Exact answer-set match rate:{mean('Exact_Set_Match'): .4f}")

    print("\n--- Partial Exact Match (PEM) ---")
    print("PEM matches when one normalized answer is a prefix of the other.")
    print(f"Mean PEM Avg:               {mean('PEM_avg'):.4f}")
    print(f"Mean Precision:             {mean('Precision_PEM'):.4f}")
    print(f"Mean Recall:                {mean('Recall_PEM'):.4f}")
    print(f"Mean F1:                    {mean('F1_PEM'):.4f}")
    print(f"Mean Accuracy:              {mean('Accuracy_PEM'):.4f}")

    print("\nDetailed results saved to:")
    print(output_path.resolve())
    print("=" * 64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate CWQ QA predictions using EM/PEM/HIT metrics adapted from Spider."
    )
    parser.add_argument("--predictions_file", required=True,
                        help="Prediction JSON/JSONL or inference output JSON with 'predictions' array")
    parser.add_argument("--gold_file", required=True,
                        help="CWQ QA test JSON, e.g. cwq_pos_test.json or cwq_neg_test.json")
    parser.add_argument("--output_file", required=True,
                        help="Output JSON path for per-question metric details")
    args = parser.parse_args()
    run_evaluation(args.predictions_file, args.gold_file, args.output_file)
