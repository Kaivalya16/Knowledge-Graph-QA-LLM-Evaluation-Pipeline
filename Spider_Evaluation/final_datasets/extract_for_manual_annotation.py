import json
import pandas as pd
import os


# ============================================================
# CONFIGURATION
# ============================================================

POSITIVE_FILE = "final_datasets/positive_all_ultimate_final.json"
NEGATIVE_FILE = "final_datasets/negative_all_ultimate_final.json"

OUTPUT_FILES = [
    "final_datasets/combined/Kaivalya.xlsx",
    "final_datasets/combined/soumajit.xlsx",
    "final_datasets/combined/vedansh.xlsx",
    "final_datasets/combined/avik.xlsx"
]

EXAMPLES_PER_FILE = 50
TOTAL_FILES = len(OUTPUT_FILES)
TOTAL_REQUIRED = EXAMPLES_PER_FILE * TOTAL_FILES

RANDOM_SEED = 42


# ============================================================
# LOAD POSITIVE DATA
# ============================================================

print("Loading positive dataset...")

with open(POSITIVE_FILE, "r", encoding="utf-8") as f:
    positive_data = json.load(f)

print(f"Positive examples: {len(positive_data)}")


# ============================================================
# LOAD NEGATIVE DATA
# ============================================================

print("Loading negative dataset...")

with open(NEGATIVE_FILE, "r", encoding="utf-8") as f:
    negative_data = json.load(f)

print(f"Negative examples: {len(negative_data)}")


# ============================================================
# CREATE LOOKUP FOR NEGATIVE EXAMPLES
# ============================================================

negative_by_id = {}

for item in negative_data:

    example_id = item.get("id")

    if example_id is not None:
        negative_by_id[example_id] = item


# ============================================================
# PAIR POSITIVE + NEGATIVE EXAMPLES
# ============================================================

paired_rows = []

missing_negative = 0

for positive in positive_data:

    example_id = positive.get("id")

    if example_id not in negative_by_id:
        missing_negative += 1
        continue

    negative = negative_by_id[example_id]

    row = {
        "id": example_id,

        "db_id": positive.get(
            "db_id",
            negative.get("db_id", "")
        ),

        "question": positive.get(
            "question",
            ""
        ),

        "negated_question": negative.get(
            "question",
            ""
        ),

        "answer": json.dumps(
            positive.get("answer", []),
            ensure_ascii=False
        ),

        "negated_answer": json.dumps(
            negative.get("answer", []),
            ensure_ascii=False
        ),

        "sql": positive.get(
            "sql",
            ""
        ),

        "negated_sql": negative.get(
            "sql",
            ""
        )
    }

    paired_rows.append(row)


print("\nPairing results:")
print(f"Successfully paired: {len(paired_rows)}")
print(f"Missing negative examples: {missing_negative}")


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(paired_rows)


# ============================================================
# REMOVE DUPLICATE IDs
# ============================================================

before = len(df)

df = df.drop_duplicates(
    subset=["id"]
).reset_index(drop=True)

after = len(df)

print(f"\nDuplicate IDs removed: {before - after}")
print(f"Unique paired examples: {after}")


# ============================================================
# CHECK WHETHER WE HAVE ENOUGH EXAMPLES
# ============================================================

if len(df) < TOTAL_REQUIRED:

    raise ValueError(
        f"\nNot enough paired examples!\n"
        f"Required: {TOTAL_REQUIRED}\n"
        f"Available: {len(df)}"
    )


# ============================================================
# RANDOMLY SELECT 200 EXAMPLES
# ============================================================

df = df.sample(
    n=TOTAL_REQUIRED,
    random_state=RANDOM_SEED
).reset_index(drop=True)


# ============================================================
# SPLIT INTO FOUR NON-OVERLAPPING GROUPS
# ============================================================

print("\nCreating splits...")

splits = []

for i in range(TOTAL_FILES):

    start = i * EXAMPLES_PER_FILE
    end = start + EXAMPLES_PER_FILE

    split = df.iloc[start:end].copy()

    splits.append(split)

    print(
        f"Split {i + 1}: "
        f"rows {start + 1} - {end}"
    )


# ============================================================
# CREATE XLSX FILES
# ============================================================

print("\nCreating Excel files...")

for i, output_file in enumerate(OUTPUT_FILES):

    person_df = splits[i].copy()

    # Shuffle rows inside the individual file
    person_df = person_df.sample(
        frac=1,
        random_state=RANDOM_SEED + i + 100
    ).reset_index(drop=True)

    # Write ONE worksheet
    person_df.to_excel(
        output_file,
        index=False,
        sheet_name="Spider QA"
    )

    print(
        f"Created: {output_file} "
        f"({len(person_df)} examples)"
    )


# ============================================================
# VERIFY OUTPUT FILES
# ============================================================

print("\n" + "=" * 70)
print("VERIFYING OUTPUT FILES")
print("=" * 70)

all_ids = []

for output_file in OUTPUT_FILES:

    if not os.path.exists(output_file):

        raise FileNotFoundError(
            f"{output_file} was not created!"
        )

    check_df = pd.read_excel(output_file)

    print(
        f"{output_file}: "
        f"{len(check_df)} examples"
    )

    # Check expected number
    if len(check_df) != EXAMPLES_PER_FILE:

        raise ValueError(
            f"{output_file} does not contain "
            f"{EXAMPLES_PER_FILE} examples!"
        )

    # Check required columns
    required_columns = [
        "id",
        "db_id",
        "question",
        "negated_question",
        "answer",
        "negated_answer",
        "sql",
        "negated_sql"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in check_df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"{output_file} is missing columns: "
            f"{missing_columns}"
        )

    all_ids.extend(
        check_df["id"].tolist()
    )


# ============================================================
# CHECK FOR DUPLICATES ACROSS FILES
# ============================================================

total_ids = len(all_ids)
unique_ids = len(set(all_ids))

duplicate_count = total_ids - unique_ids

print("\n" + "=" * 70)
print("DUPLICATE CHECK")
print("=" * 70)

print(f"Total examples:  {total_ids}")
print(f"Unique examples: {unique_ids}")
print(f"Duplicates:      {duplicate_count}")


if duplicate_count == 0:

    print(
        "\nSUCCESS!"
        "\nAll four Excel files contain completely "
        "different examples."
    )

else:

    raise ValueError(
        f"\nERROR: {duplicate_count} duplicate examples "
        f"found across the files!"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

for output_file in OUTPUT_FILES:

    print(
        f"{output_file:<20} -> "
        f"{EXAMPLES_PER_FILE} examples"
    )

print(
    f"\nTotal: {TOTAL_REQUIRED} unique examples"
)

print("\nDone!")