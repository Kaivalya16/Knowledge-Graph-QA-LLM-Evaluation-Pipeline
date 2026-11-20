import pandas as pd
import ast
import re

# ============================================================
# INPUT FILES
# ============================================================

QA_UNMATCHED = "data/final/qa_dataset_unmatched.csv"
COMPARATIVE_UNMATCHED = "data/final/unmatched_comparative.csv"

# Output
OUTPUT = "data/qa_dataset_comparative.csv"


# ============================================================
# NORMALIZATION FUNCTIONS
# ============================================================

def normalize_text(x):
    if pd.isna(x):
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(x).strip().lower()
    )


def normalize_answers(x):
    """
    Convert answers into a comparable representation.
    """

    try:
        obj = ast.literal_eval(str(x))

        if isinstance(obj, list):
            values = []

            for item in obj:

                if isinstance(item, dict):
                    values.append(
                        str(item.get("answer", "")).strip().lower()
                    )
                else:
                    values.append(
                        str(item).strip().lower()
                    )

            return tuple(sorted(values))

        return (str(obj).strip().lower(),)

    except Exception:
        return (
            str(x).strip().lower(),
        )


# ============================================================
# LOAD FILES
# ============================================================

print("Loading files...")

qa = pd.read_csv(
    QA_UNMATCHED,
    encoding="utf-8-sig"
)

comparative = pd.read_csv(
    COMPARATIVE_UNMATCHED,
    encoding="utf-8-sig"
)

print(f"QA unmatched rows          : {len(qa)}")
print(f"Comparative unmatched rows: {len(comparative)}")


# ============================================================
# CHECK ROW COUNTS
# ============================================================

if len(qa) != len(comparative):
    raise ValueError(
        "The two files do not contain the same number of rows."
    )


# ============================================================
# VERIFY ANSWERS
# ============================================================

qa_answers = qa["answers"].apply(normalize_answers)

comparative_answers = (
    comparative["answers"]
    .apply(normalize_answers)
)

answer_matches = (
    qa_answers == comparative_answers
)

print(
    f"\nAnswer matches: "
    f"{answer_matches.sum()} / {len(qa)}"
)


# ============================================================
# VERIFY SPARQL
# ============================================================

qa_sparql = (
    qa["sparql"]
    .apply(normalize_text)
)

comparative_sparql = (
    comparative["sparql"]
    .apply(normalize_text)
)

sparql_matches = (
    qa_sparql == comparative_sparql
)

print(
    f"SPARQL matches: "
    f"{sparql_matches.sum()} / {len(qa)}"
)


# ============================================================
# SAFETY CHECK
# ============================================================

if not answer_matches.all():
    raise ValueError(
        "Answers do not match for every row. "
        "Stopping to prevent incorrect extraction."
    )

if not sparql_matches.all():
    raise ValueError(
        "SPARQL does not match for every row. "
        "Stopping to prevent incorrect extraction."
    )

print("\n✓ All 1531 rows verified successfully.")


# ============================================================
# RESTORE QUESTIONS
# ============================================================

qa["original_question"] = (
    comparative["question"]
)

qa["negated_question"] = (
    comparative["negated_question"]
)


# ============================================================
# SAVE
# ============================================================

qa.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nSaved corrected dataset to:")
print(f"  {OUTPUT}")

print(f"\nTotal Comparative QA examples: {len(qa)}")