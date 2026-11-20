import json
import csv
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "final_datasets/negative_all_ultimate_final.json"
OUTPUT_FILE = "final_datasets/negative_all_ultimate_final.csv"

# ============================================================
# LOAD JSON
# ============================================================

print("Loading JSON...")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total records: {len(data)}")

# ============================================================
# CSV COLUMNS
# ============================================================

fieldnames = [
    "id",
    "source_dataset",
    "db_id",
    "question",
    "sql",
    "answer",
    "answer_count",
    "negation_type"
]

# ============================================================
# CONVERT TO CSV
# ============================================================

print("Converting to CSV...")

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for item in data:

        # Keep nested answer structure intact
        answer = item.get("answer", [])

        answer_string = json.dumps(
            answer,
            ensure_ascii=False
        )

        writer.writerow({
            "id": item.get("id", ""),
            "source_dataset": item.get("source_dataset", ""),
            "db_id": item.get("db_id", ""),
            "question": item.get("question", ""),
            "sql": item.get("sql", ""),
            "answer": answer_string,
            "answer_count": item.get("answer_count", ""),
            "negation_type": item.get("negation_type", "")
        })

# ============================================================
# DONE
# ============================================================

print("\n========================================")
print("Conversion completed!")
print(f"Records : {len(data)}")
print(f"Output  : {OUTPUT_FILE}")
print("========================================")