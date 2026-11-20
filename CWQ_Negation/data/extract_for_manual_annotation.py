import pandas as pd
import os

# ============================================================
# INPUT FILES
# ============================================================

comparative_file = "data/qa_dataset_comparative.csv"
conjunction_file = "data/qa_dataset_conjunction.csv"
composition_file = "data/qa_dataset_composition.csv"

# Output files
output_files = [
    "data/annotate/Kaivalya.xlsx",
    "data/annotate/soumajit.xlsx",
    "data/annotate/vedansh.xlsx",
    "data/annotate/avik.xlsx"
]

# ============================================================
# REQUIRED NUMBER OF EXAMPLES
# ============================================================

N_COMPARATIVE = 16
N_CONJUNCTION = 16
N_COMPOSITION = 18

RANDOM_SEED = 42


# ============================================================
# READ DATASETS
# ============================================================

comparative = pd.read_csv(comparative_file)
conjunction = pd.read_csv(conjunction_file)
composition = pd.read_csv(composition_file)

print("Original dataset sizes:")
print("Comparative :", len(comparative))
print("Conjunction :", len(conjunction))
print("Composition :", len(composition))


# ============================================================
# REMOVE DUPLICATE ROWS
# ============================================================

comparative = comparative.drop_duplicates().reset_index(drop=True)
conjunction = conjunction.drop_duplicates().reset_index(drop=True)
composition = composition.drop_duplicates().reset_index(drop=True)


# ============================================================
# CHECK WHETHER ENOUGH UNIQUE EXAMPLES EXIST
# ============================================================

if len(comparative) < N_COMPARATIVE * 4:
    raise ValueError(
        f"Not enough unique comparative examples. "
        f"Required: {N_COMPARATIVE * 4}, "
        f"Available: {len(comparative)}"
    )

if len(conjunction) < N_CONJUNCTION * 4:
    raise ValueError(
        f"Not enough unique conjunction examples. "
        f"Required: {N_CONJUNCTION * 4}, "
        f"Available: {len(conjunction)}"
    )

if len(composition) < N_COMPOSITION * 4:
    raise ValueError(
        f"Not enough unique composition examples. "
        f"Required: {N_COMPOSITION * 4}, "
        f"Available: {len(composition)}"
    )


# ============================================================
# SHUFFLE EACH CATEGORY
# ============================================================

comparative = comparative.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)

conjunction = conjunction.sample(
    frac=1,
    random_state=RANDOM_SEED + 1
).reset_index(drop=True)

composition = composition.sample(
    frac=1,
    random_state=RANDOM_SEED + 2
).reset_index(drop=True)


# ============================================================
# TAKE ONLY THE 64 + 64 + 72 REQUIRED EXAMPLES
# ============================================================

comparative = comparative.iloc[:N_COMPARATIVE * 4].copy()
conjunction = conjunction.iloc[:N_CONJUNCTION * 4].copy()
composition = composition.iloc[:N_COMPOSITION * 4].copy()


# ============================================================
# ADD CATEGORY COLUMN
# ============================================================

comparative["category"] = "comparative"
conjunction["category"] = "conjunction"
composition["category"] = "composition"


# ============================================================
# SPLIT INTO 4 GROUPS
# ============================================================

comparative_splits = [
    comparative.iloc[
        i * N_COMPARATIVE:(i + 1) * N_COMPARATIVE
    ].copy()
    for i in range(4)
]

conjunction_splits = [
    conjunction.iloc[
        i * N_CONJUNCTION:(i + 1) * N_CONJUNCTION
    ].copy()
    for i in range(4)
]

composition_splits = [
    composition.iloc[
        i * N_COMPOSITION:(i + 1) * N_COMPOSITION
    ].copy()
    for i in range(4)
]


# ============================================================
# CREATE ONE XLSX FILE PER PERSON
# ============================================================

for i, output_file in enumerate(output_files):

    # Combine all 3 categories into ONE dataframe
    person_data = pd.concat(
        [
            comparative_splits[i],
            conjunction_splits[i],
            composition_splits[i]
        ],
        ignore_index=True
    )

    # Optional: shuffle the 50 rows so categories aren't grouped
    person_data = person_data.sample(
        frac=1,
        random_state=RANDOM_SEED + 100 + i
    ).reset_index(drop=True)

    # Write ONE worksheet only
    person_data.to_excel(
        output_file,
        index=False,
        sheet_name="QA Dataset"
    )

    print(f"\nCreated: {output_file}")
    print(f"Total examples: {len(person_data)}")
    print(
        person_data["category"]
        .value_counts()
        .to_dict()
    )


# ============================================================
# VERIFY NO OVERLAP BETWEEN FILES
# ============================================================

print("\n" + "=" * 60)
print("VERIFYING NO OVERLAP")
print("=" * 60)

all_data = []

for output_file in output_files:

    df = pd.read_excel(output_file)

    print(
        f"{output_file}: "
        f"{len(df)} rows | "
        f"{df['category'].value_counts().to_dict()}"
    )

    all_data.append(df)


# Combine all four files
combined = pd.concat(all_data, ignore_index=True)

# Check duplicate complete rows
duplicate_rows = combined.duplicated().sum()

if duplicate_rows == 0:
    print("\nSUCCESS: No duplicate examples across the 4 files.")
else:
    print(
        f"\nWARNING: {duplicate_rows} duplicate examples found!"
    )


print("\nDone!")