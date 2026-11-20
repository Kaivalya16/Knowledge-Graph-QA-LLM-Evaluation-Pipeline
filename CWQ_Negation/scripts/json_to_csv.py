#!/usr/bin/env python3
"""
Convert a JSON file to CSV.

Handles:
  - A list of objects:          [{"a": 1}, {"a": 2}]
  - A single object:            {"a": 1, "b": 2}
  - An object wrapping a list:  {"data": [{"a": 1}, ...]}   (auto-detected, or use --key)
  - JSON Lines (one object per line), with --jsonl
  - Nested objects, flattened:  {"user": {"name": "x"}} -> column "user.name"
  - Lists inside records are written as JSON strings in the cell

Usage:
  python json_to_csv.py input.json                 # writes input.csv
  python json_to_csv.py input.json -o output.csv
  python json_to_csv.py input.json --key results   # use the list under "results"
  python json_to_csv.py input.jsonl --jsonl
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def flatten(obj, parent_key="", sep="."):
    """Flatten nested dicts into a single-level dict with dotted keys."""
    items = {}
    for key, value in obj.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
        if isinstance(value, dict):
            items.update(flatten(value, new_key, sep))
        elif isinstance(value, list):
            items[new_key] = json.dumps(value, ensure_ascii=False)
        else:
            items[new_key] = value
    return items


def load_records(path, key=None, jsonl=False):
    """Read the file and return a list of records (dicts)."""
    with open(path, encoding="utf-8") as f:
        if jsonl:
            data = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)

    if key:
        for part in key.split("."):
            data = data[part]
    elif isinstance(data, dict):
        # If the object holds exactly one list of objects, use that list
        lists = [v for v in data.values()
                 if isinstance(v, list) and v and all(isinstance(i, dict) for i in v)]
        data = lists[0] if len(lists) == 1 else [data]

    if not isinstance(data, list):
        data = [data]

    # Wrap plain values (e.g. [1, 2, 3]) so every record is a dict
    return [r if isinstance(r, dict) else {"value": r} for r in data]


def json_to_csv(input_path, output_path, key=None, jsonl=False, sep="."):
    records = [flatten(r, sep=sep) for r in load_records(input_path, key, jsonl)]
    if not records:
        sys.exit("No records found in the JSON file.")

    # Collect every column, keeping first-seen order
    headers = list(dict.fromkeys(k for r in records for k in r))

    # utf-8-sig so Excel opens non-English text correctly
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers, restval="")
        writer.writeheader()
        writer.writerows(records)

    print(f"Wrote {len(records)} rows x {len(headers)} columns to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert a JSON file to CSV.")
    parser.add_argument("input", help="Path to the JSON file")
    parser.add_argument("-o", "--output", help="Output CSV path (default: same name, .csv)")
    parser.add_argument("--key", help="Dotted path to the list of records, e.g. 'data.items'")
    parser.add_argument("--jsonl", action="store_true", help="Input is JSON Lines")
    parser.add_argument("--sep", default=".", help="Separator for nested keys (default '.')")
    args = parser.parse_args()

    output = args.output or str(Path(args.input).with_suffix(".csv"))
    json_to_csv(args.input, output, args.key, args.jsonl, args.sep)


if __name__ == "__main__":
    main()