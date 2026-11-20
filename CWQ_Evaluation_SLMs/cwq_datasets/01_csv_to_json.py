#!/usr/bin/env python3
"""
Convert a CSV file to a JSON file.

Each CSV row becomes a JSON object, using the header row as keys.

Usage:
    python csv_to_json.py input.csv
    python csv_to_json.py input.csv -o output.json
    python csv_to_json.py input.csv --delimiter ";" --indent 2 --convert-types
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def convert_value(value):
    """Turn numeric/boolean/empty strings into proper JSON types."""
    if value is None:
        return None
    v = value.strip()
    if v == "":
        return None
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return value


def csv_to_json(input_path, output_path, delimiter=",", indent=4,
                encoding="utf-8-sig", convert_types=False):
    with open(input_path, mode="r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        rows = []
        for row in reader:
            if convert_types:
                row = {k: convert_value(v) for k, v in row.items()}
            rows.append(row)

    with open(output_path, mode="w", encoding="utf-8") as f:
        json.dump(rows, f, indent=indent, ensure_ascii=False)

    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Convert a CSV file to JSON.")
    parser.add_argument("input", help="Path to the input .csv file")
    parser.add_argument("-o", "--output",
                        help="Path to the output .json file (default: same name as input)")
    parser.add_argument("-d", "--delimiter", default=",",
                        help="CSV delimiter (default: ',')")
    parser.add_argument("-i", "--indent", type=int, default=4,
                        help="JSON indentation spaces (default: 4)")
    parser.add_argument("-e", "--encoding", default="utf-8-sig",
                        help="Input file encoding (default: utf-8-sig, handles BOM)")
    parser.add_argument("-t", "--convert-types", action="store_true",
                        help="Convert numbers, booleans and empty values to JSON types")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        sys.exit(f"Error: file not found: {input_path}")

    output_path = Path(args.output) if args.output else input_path.with_suffix(".json")

    try:
        count = csv_to_json(input_path, output_path, args.delimiter,
                            args.indent, args.encoding, args.convert_types)
    except (csv.Error, UnicodeDecodeError) as exc:
        sys.exit(f"Error reading CSV: {exc}")

    print(f"Converted {count} rows: {input_path} -> {output_path}")


if __name__ == "__main__":
    main()