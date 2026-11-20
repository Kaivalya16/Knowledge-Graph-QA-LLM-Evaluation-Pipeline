#!/usr/bin/env python3
"""
Split a CWQ JSON dataset into train and test sets.

Supports:
  - A JSON array: [ {...}, {...} ]
  - A JSON object containing a list under a common key such as "data" or "examples"
  - JSONL: one JSON object per line

The split is performed at the record level. Each record keeps its original_question,
negated_question, answers, negated_answers, SPARQL fields, and all other fields
together, preventing positive/negative versions of one record from crossing splits.
"""

import argparse
import json
import random
from pathlib import Path


def load_records(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Input file is empty: {path}")

    # Try regular JSON first.
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Fall back to JSONL.
        records = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Could not parse JSON or JSONL (error on line {line_no}): {exc}"
                ) from exc
            if not isinstance(item, dict):
                raise ValueError(f"JSONL line {line_no} is not a JSON object.")
            records.append(item)
        return records

    if isinstance(obj, list):
        records = obj
    elif isinstance(obj, dict):
        # Accept a common wrapper format, otherwise treat this as one record.
        records = None
        for key in ("data", "examples", "records", "questions"):
            if isinstance(obj.get(key), list):
                records = obj[key]
                break
        if records is None:
            records = [obj]
    else:
        raise ValueError("Top-level JSON must be a list or an object.")

    if not all(isinstance(item, dict) for item in records):
        raise ValueError("Every dataset record must be a JSON object.")
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Split a CWQ JSON dataset into reproducible train/test files."
    )
    parser.add_argument("--input_file", required=True, help="Path to the CWQ .json or .jsonl file")
    parser.add_argument("--output_dir", default="cwq_splits", help="Directory for output files")
    parser.add_argument("--test_size", type=float, default=0.20, help="Fraction for test set (default: 0.20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    if not 0 < args.test_size < 1:
        parser.error("--test_size must be greater than 0 and less than 1")

    input_path = Path(args.input_file)
    if not input_path.is_file():
        parser.error(f"Input file not found: {input_path}")

    records = load_records(input_path)
    n = len(records)
    if n < 2:
        parser.error(f"Need at least 2 records to split; found {n}")

    # Report missing IDs/fields, but preserve all records without silently dropping data.
    required = ("id", "original_question", "negated_question", "answers", "negated_answers")
    missing_counts = {
        field: sum(field not in record or record[field] is None for record in records)
        for field in required
    }

    indices = list(range(n))
    random.Random(args.seed).shuffle(indices)

    n_test = max(1, round(n * args.test_size))
    n_test = min(n - 1, n_test)
    test_indices = set(indices[:n_test])

    train_records = [record for i, record in enumerate(records) if i not in test_indices]
    test_records = [record for i, record in enumerate(records) if i in test_indices]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "cwq_train.json"
    test_path = output_dir / "cwq_test.json"

    train_path.write_text(
        json.dumps(train_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    test_path.write_text(
        json.dumps(test_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Verify record IDs do not overlap where IDs are available.
    train_ids = {str(r["id"]) for r in train_records if r.get("id") is not None}
    test_ids = {str(r["id"]) for r in test_records if r.get("id") is not None}
    overlap = train_ids & test_ids

    print("\nCWQ split completed")
    print("-" * 48)
    print(f"Input records : {n}")
    print(f"Train records : {len(train_records)} ({len(train_records) / n:.1%})")
    print(f"Test records  : {len(test_records)} ({len(test_records) / n:.1%})")
    print(f"Random seed   : {args.seed}")
    print(f"Train output  : {train_path.resolve()}")
    print(f"Test output   : {test_path.resolve()}")
    print(f"ID overlap    : {len(overlap)}")
    if overlap:
        print("WARNING: IDs appear in both splits. Check whether the source has duplicate IDs.")
    print("\nRecords missing required fields (preserved, not dropped):")
    for field, count in missing_counts.items():
        print(f"  {field}: {count}")


if __name__ == "__main__":
    main()
