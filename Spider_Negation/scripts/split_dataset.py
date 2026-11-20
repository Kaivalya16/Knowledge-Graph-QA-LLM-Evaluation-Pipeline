import json
import random
from pathlib import Path

# -----------------------------
# Configuration
# -----------------------------
INPUT_FILE = "final_datasets_checked/negative_strict.json"
OUTPUT_DIR = "final_datasets_checked"

TRAIN_RATIO = 0.7
RANDOM_SEED = 42

# -----------------------------
# Create output directory
# -----------------------------
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# -----------------------------
# Load dataset
# -----------------------------
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"Loaded {len(dataset)} samples.")

# -----------------------------
# Shuffle
# -----------------------------
random.seed(RANDOM_SEED)
random.shuffle(dataset)

# -----------------------------
# Split
# -----------------------------
split_index = int(TRAIN_RATIO * len(dataset))

train_set = dataset[:split_index]
test_set = dataset[split_index:]

print(f"Train samples : {len(train_set)}")
print(f"Test samples  : {len(test_set)}")

# -----------------------------
# Save
# -----------------------------
with open(f"{OUTPUT_DIR}/neg_train.json", "w", encoding="utf-8") as f:
    json.dump(train_set, f, indent=4, ensure_ascii=False)

with open(f"{OUTPUT_DIR}/neg_test.json", "w", encoding="utf-8") as f:
    json.dump(test_set, f, indent=4, ensure_ascii=False)

print("Done.")