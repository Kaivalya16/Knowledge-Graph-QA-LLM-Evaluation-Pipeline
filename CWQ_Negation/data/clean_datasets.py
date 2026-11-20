import pandas as pd
import os

# ============================================================
# INPUT FILES
# ============================================================

FILES = [
    "data/qa_dataset_comparative.csv",
    "data/qa_dataset_composition.csv",
    "data/qa_dataset_conjunction.csv"
]


# ============================================================
# PROCESS EACH FILE
# ============================================================

for file in FILES:

    print("\n" + "=" * 70)
    print(f"Processing: {file}")
    print("=" * 70)

    df = pd.read_csv(
        file,
        encoding="utf-8-sig"
    )

    original_count = len(df)

    print(f"Original examples: {original_count}")

    # --------------------------------------------------------
    # Treat the following as empty:
    #   NaN
    #   empty string
    #   whitespace
    # --------------------------------------------------------

    df = df.replace(r"^\s*$", pd.NA, regex=True)

    # --------------------------------------------------------
    # Remove rows where ANY field is empty
    # --------------------------------------------------------

    empty_mask = df.isna().any(axis=1)

    removed = empty_mask.sum()

    cleaned_df = df[~empty_mask].copy()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = file.replace(
        ".csv",
        "_cleaned.csv"
    )

    cleaned_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print(f"Removed examples : {removed}")
    print(f"Remaining examples: {len(cleaned_df)}")
    print(f"Saved to: {output_file}")


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("CLEANING COMPLETE")
print("=" * 70)