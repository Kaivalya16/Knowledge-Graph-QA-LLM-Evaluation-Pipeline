import json
import pandas as pd
import re


# ============================================================
# FILE PATHS
# ============================================================

QA_JSON = "data/qa_dataset.json"

COMPOSITION_CSV = "data/final compsition question neg.csv"
CONJUNCTION_CSV = "data/final conjunction question neg.csv"
COMPARATIVE_CSV = "data/final comparative question negated.csv"


# Output files
COMPOSITION_OUT = "data/final/qa_dataset_composition.csv"
CONJUNCTION_OUT = "data/final/qa_dataset_conjunction.csv"
COMPARATIVE_OUT = "data/final/qa_dataset_comparative.csv"

# QA examples from JSON that did not match ANY category
UNMATCHED_QA_OUT = "data/final/qa_dataset_unmatched.csv"

# Questions from each category CSV that did not match JSON
UNMATCHED_COMPOSITION_OUT = "data/final/unmatched_composition.csv"
UNMATCHED_CONJUNCTION_OUT = "data/final/unmatched_conjunction.csv"
UNMATCHED_COMPARATIVE_OUT = "data/final/unmatched_comparative.csv"


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_question(text):
    if pd.isna(text):
        return ""

    text = str(text).strip().lower()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text


# ============================================================
# LOAD QA DATASET
# ============================================================

print("Loading qa_dataset.json...")

with open(QA_JSON, "r", encoding="utf-8") as f:
    qa_data = json.load(f)

qa_df = pd.DataFrame(qa_data)

print(f"Total QA examples: {len(qa_df)}")

if "negated_question" not in qa_df.columns:
    raise ValueError(
        "qa_dataset.json does not contain 'negated_question'"
    )

qa_df["_match_question"] = (
    qa_df["negated_question"]
    .apply(normalize_question)
)


# ============================================================
# FUNCTION
# ============================================================

def process_category(csv_file, matched_output, unmatched_output):

    print("\n" + "=" * 70)
    print(f"Processing: {csv_file}")
    print("=" * 70)

    csv_df = pd.read_csv(csv_file)

    if "negated_question" not in csv_df.columns:
        raise ValueError(
            f"{csv_file} does not contain 'negated_question'"
        )

    print(f"Questions in CSV: {len(csv_df)}")

    # --------------------------------------------------------
    # Normalize CSV questions
    # --------------------------------------------------------

    csv_df["_match_question"] = (
        csv_df["negated_question"]
        .apply(normalize_question)
    )

    # --------------------------------------------------------
    # Questions which exist in JSON
    # --------------------------------------------------------

    matched_questions = set(
        csv_df["_match_question"]
    ) & set(
        qa_df["_match_question"]
    )

    # --------------------------------------------------------
    # Extract complete QA examples from JSON
    # --------------------------------------------------------

    matched_qa = qa_df[
        qa_df["_match_question"].isin(matched_questions)
    ].copy()

    matched_qa.drop(
        columns=["_match_question"],
        inplace=True
    )

    matched_qa.to_csv(
        matched_output,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Extract CSV questions which DON'T exist in JSON
    # --------------------------------------------------------

    unmatched_csv = csv_df[
        ~csv_df["_match_question"].isin(
            set(qa_df["_match_question"])
        )
    ].copy()

    unmatched_csv.drop(
        columns=["_match_question"],
        inplace=True
    )

    unmatched_csv.to_csv(
        unmatched_output,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print(f"Unique questions in CSV : {csv_df['_match_question'].nunique()}")
    print(f"Matched QA examples     : {len(matched_qa)}")
    print(f"Unmatched CSV questions : {len(unmatched_csv)}")

    print(f"\nMatched saved to:")
    print(f"  {matched_output}")

    print(f"Unmatched saved to:")
    print(f"  {unmatched_output}")

    return matched_questions


# ============================================================
# PROCESS ALL THREE CATEGORIES
# ============================================================

composition_matches = process_category(
    COMPOSITION_CSV,
    COMPOSITION_OUT,
    UNMATCHED_COMPOSITION_OUT
)

conjunction_matches = process_category(
    CONJUNCTION_CSV,
    CONJUNCTION_OUT,
    UNMATCHED_CONJUNCTION_OUT
)

comparative_matches = process_category(
    COMPARATIVE_CSV,
    COMPARATIVE_OUT,
    UNMATCHED_COMPARATIVE_OUT
)


# ============================================================
# FIND QA EXAMPLES THAT MATCH NONE OF THE THREE CSVs
# ============================================================

all_matched_questions = (
    composition_matches
    | conjunction_matches
    | comparative_matches
)

unmatched_qa = qa_df[
    ~qa_df["_match_question"].isin(all_matched_questions)
].copy()

unmatched_qa.drop(
    columns=["_match_question"],
    inplace=True
)

unmatched_qa.to_csv(
    UNMATCHED_QA_OUT,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"Total QA examples in JSON : {len(qa_df)}")

print(f"Composition matched       : {len(composition_matches)}")
print(f"Conjunction matched       : {len(conjunction_matches)}")
print(f"Comparative matched       : {len(comparative_matches)}")

print(
    f"Total matched             : "
    f"{len(all_matched_questions)}"
)

print(
    f"Unmatched QA examples     : "
    f"{len(unmatched_qa)}"
)

print("\nOutput files:")
print(f"  {COMPOSITION_OUT}")
print(f"  {CONJUNCTION_OUT}")
print(f"  {COMPARATIVE_OUT}")

print("\nUnmatched from JSON:")
print(f"  {UNMATCHED_QA_OUT}")

print("\nUnmatched from category CSVs:")
print(f"  {UNMATCHED_COMPOSITION_OUT}")
print(f"  {UNMATCHED_CONJUNCTION_OUT}")
print(f"  {UNMATCHED_COMPARATIVE_OUT}")

print("\nDone.")