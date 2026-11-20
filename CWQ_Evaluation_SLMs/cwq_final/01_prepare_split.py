#!/usr/bin/env python3
"""
Create four standard QA datasets from CWQ train/test split files.

Input files:
  cwq_train.json
  cwq_test.json

Output files:
  cwq_pos_train.json
  cwq_pos_test.json
  cwq_neg_train.json
  cwq_neg_test.json

Positive examples use original_question -> answers.
Negative examples use negated_question -> negated_answers.

Answer values are normalized into JSON arrays. The script supports answers already
stored as arrays, Python-list-like strings (single or double quoted), and plain
strings. It does not use eval().
"""

import argparse
import ast
import json
from pathlib import Path


def load_records(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Input file is empty: {path}")

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Support JSONL files too.
        records = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON/JSONL at line {line_no} in {path}: {exc}"
                ) from exc
            if not isinstance(item, dict):
                raise ValueError(f"Line {line_no} in {path} is not a JSON object.")
            records.append(item)
        return records

    if isinstance(obj, list):
        records = obj
    elif isinstance(obj, dict):
        records = None
        for key in ("data", "examples", "records", "questions"):
            if isinstance(obj.get(key), list):
                records = obj[key]
                break
        if records is None:
            records = [obj]
    else:
        raise ValueError(f"Unsupported top-level JSON type in {path}")

    if not all(isinstance(item, dict) for item in records):
        raise ValueError(f"Every record in {path} must be a JSON object.")
    return records


def normalize_answers(value):
    """Return answer values as a list without executing arbitrary code."""
    if value is None:
        return []

    if isinstance(value, list):
        return [str(v).strip() for v in value if v is not None and str(v).strip()]

    if isinstance(value, tuple):
        return [str(v).strip() for v in value if v is not None and str(v).strip()]

    if not isinstance(value, str):
        value = str(value)

    value = value.strip()
    if not value:
        return []

    # Parse strings such as "['Eclipse']" or '["Eclipse", "Hard Candy"]'.
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple, set)):
            return [str(v).strip() for v in parsed if v is not None and str(v).strip()]
        if isinstance(parsed, str):
            return [parsed.strip()] if parsed.strip() else []
    except (ValueError, SyntaxError):
        pass

    # If it was not a list-like representation, treat it as one answer.
    return [value]


def make_qa_records(records, question_key, answer_key, split_name, case_name):
    output = []
    missing_question = 0
    missing_answer = 0
    empty_answer = 0

    for index, record in enumerate(records):
        question = record.get(question_key)
        if not isinstance(question, str) or not question.strip():
            missing_question += 1
            continue

        if answer_key not in record or record.get(answer_key) is None:
            missing_answer += 1
            answers = []
        else:
            answers = normalize_answers(record.get(answer_key))

        if not answers:
            empty_answer += 1

        # Keep a stable source identifier to trace predictions back to the CWQ row.
        qa_record = {
            "id": str(record.get("id", f"{split_name}_{index}")),
            "question": question.strip(),
            "answers": answers,
        }
        output.append(qa_record)

    stats = {
        "input_records": len(records),
        "output_records": len(output),
        "missing_question_skipped": missing_question,
        "missing_answer_field": missing_answer,
        "empty_answer_list": empty_answer,
    }
    return output, stats


def write_json(path: Path, records):
    path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build positive and negative CWQ QA datasets from train/test splits."
    )
    parser.add_argument("--train_file", required=True, help="Path to cwq_train.json")
    parser.add_argument("--test_file", required=True, help="Path to cwq_test.json")
    parser.add_argument("--output_dir", default="data/qa_datasets", help="Output directory")
    args = parser.parse_args()

    train_path = Path(args.train_file)
    test_path = Path(args.test_file)
    for path in (train_path, test_path):
        if not path.is_file():
            parser.error(f"File not found: {path}")

    train_records = load_records(train_path)
    test_records = load_records(test_path)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("cwq_pos_train.json", train_records, "original_question", "answers", "train", "positive"),
        ("cwq_pos_test.json", test_records, "original_question", "answers", "test", "positive"),
        ("cwq_neg_train.json", train_records, "negated_question", "negated_answers", "train", "negative"),
        ("cwq_neg_test.json", test_records, "negated_question", "negated_answers", "test", "negative"),
    ]

    print("\nCWQ QA dataset conversion")
    print("=" * 64)
    for filename, records, qkey, akey, split, case in configs:
        qa_records, stats = make_qa_records(records, qkey, akey, split, case)
        out_path = out_dir / filename
        write_json(out_path, qa_records)
        print(f"\n{filename}")
        print(f"  {len(qa_records)} records -> {out_path.resolve()}")
        print(f"  Missing questions skipped : {stats['missing_question_skipped']}")
        print(f"  Missing answer fields     : {stats['missing_answer_field']}")
        print(f"  Empty answer lists        : {stats['empty_answer_list']}")

    print("\nEach output record has this structure:")
    print(json.dumps({
        "id": "3003",
        "question": "What movie directed by David Slade did Robert Pattinson star in",
        "answers": ["Eclipse"]
    }, ensure_ascii=False, indent=2))
    print("\nNote: answer lists are normalized from strings like \"['Eclipse']\" into JSON arrays.")


if __name__ == "__main__":
    main()
