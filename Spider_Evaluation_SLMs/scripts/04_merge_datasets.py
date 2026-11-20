"""
Stage 4 (v2): merge the per-split outputs of Stage 3 into one dataset.

Reads, for train_spider, train_others and dev:

    data/processed/final/positive_<split>.json
    data/processed/final/negative_<split>.json
    data/processed/final/validated_pairs_<split>.json

Writes:

    final_datasets/positive_all.json   positive QA, one row per pair
    final_datasets/negative_all.json   negative QA, same order and ids
    final_datasets/pairs_all.json      full pairs (both sides + metadata)
    final_datasets/native_negatives_all.json
                                       natively negative Spider questions that
                                       Stage 1 set aside (no positive partner);
                                       Stage 5 adds them to the negative data
    final_datasets/stage4_report.json  counts and the pipeline funnel so far

Native negatives are read from
    data/processed/canonical_positive/native_negative_<split>.json

No question is filtered here. Changes from v1:

  1. Integrity checks instead of a length comparison: ids must be unique,
     every negative must have its positive (same id, same position), and
     every pair must have both answers non-empty. The script stops with an
     error if a check fails, rather than writing a misaligned dataset.
  2. The merged pairs file is written too, so later stages can work on
     complete pairs instead of re-joining positive and negative by id.
  3. Splits that produced nothing (train_others, whose databases are empty)
     are reported as such instead of being silently skipped.
  4. A funnel built from the Stage 1-3 reports shows how many questions are
     left after each stage, per split, with percentages.
"""

import os
import json
import collections


# ============================================================
# CONFIGURATION
# ============================================================

SPLITS = ["train_spider", "train_others", "dev"]

INPUT_DIR = "data/processed/final"
NATIVE_DIR = "data/processed/canonical_positive"
OUTPUT_DIR = "datasets"

STAGE_REPORTS = {
    "stage1": "data/processed/canonical_positive/stage1_report.json",
    "stage2": "data/processed/negation_candidates/stage2_report.json",
    "stage3": "data/processed/final/stage3_report.json",
}

POSITIVE_OUTPUT = os.path.join(OUTPUT_DIR, "positive_all.json")
NEGATIVE_OUTPUT = os.path.join(OUTPUT_DIR, "negative_all.json")
PAIRS_OUTPUT = os.path.join(OUTPUT_DIR, "pairs_all.json")
NATIVE_OUTPUT = os.path.join(OUTPUT_DIR, "native_negatives_all.json")
REPORT_FILE = os.path.join(OUTPUT_DIR, "stage4_report.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def check_integrity(positives, negatives, pairs):
    """Return a list of problems; empty when the merged data is consistent."""
    problems = []

    if not (len(positives) == len(negatives) == len(pairs)):
        problems.append(
            f"length mismatch: positive={len(positives)} "
            f"negative={len(negatives)} pairs={len(pairs)}"
        )

    for name, rows in (("positive", positives), ("negative", negatives), ("pairs", pairs)):
        ids = [row.get("id") for row in rows]
        duplicates = [i for i, n in collections.Counter(ids).items() if n > 1]
        if duplicates:
            problems.append(f"{name}: {len(duplicates)} duplicate ids, e.g. {duplicates[:3]}")
        if any(i is None for i in ids):
            problems.append(f"{name}: rows without an id")

    for index, (positive, negative) in enumerate(zip(positives, negatives)):
        if positive.get("id") != negative.get("id"):
            problems.append(
                f"row {index}: positive id {positive.get('id')} != "
                f"negative id {negative.get('id')}"
            )
            break

    for pair in pairs:
        if not pair.get("positive_answer") or not pair.get("negative_answer"):
            problems.append(f"pair {pair.get('id')}: empty answer")
            break

    return problems


# ============================================================
# FUNNEL FROM STAGE REPORTS
# ============================================================

def build_funnel(merged_counts):
    """
    Questions left after each stage, per split.

    Stage 2 is counted as questions that produced at least one negation
    candidate; Stage 3 and Stage 4 as accepted pairs.
    """
    reports = {name: load_json(path, {}) for name, path in STAGE_REPORTS.items()}
    rows = []

    def stage_row(label, getter):
        counts = {}
        for split in SPLITS:
            try:
                counts[split] = getter(split)
            except (KeyError, TypeError):
                counts[split] = None
        rows.append({"stage": label, "counts": counts})

    stage_row("0. Original Spider",
              lambda s: reports["stage1"][s]["stats"]["total"])
    stage_row("1. Canonical positives",
              lambda s: reports["stage1"][s]["stats"]["kept"])
    stage_row("2. Questions with >=1 negation candidate",
              lambda s: reports["stage2"][s]["questions"].get("questions_with_candidate", 0))
    stage_row("3. Validated pairs (one per question)",
              lambda s: reports["stage3"][s]["questions"].get("accepted", 0))
    stage_row("4. Merged pairs",
              lambda s: merged_counts.get(s, 0))

    original_total = None
    previous_total = None
    for row in rows:
        values = [v for v in row["counts"].values() if v is not None]
        total = sum(values) if values else None
        row["total"] = total
        if total is not None:
            if original_total is None:
                original_total = total
            row["percent_of_previous"] = (
                round(100 * total / previous_total, 1) if previous_total else None
            )
            row["percent_of_original"] = (
                round(100 * total / original_total, 1) if original_total else None
            )
            previous_total = total
    return rows


def print_funnel(rows):
    header = f"{'Stage':<44}" + "".join(f"{s:>14}" for s in SPLITS) + \
        f"{'Total':>9}{'% prev':>9}{'% orig':>9}"
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = "".join(
            f"{('-' if row['counts'][s] is None else row['counts'][s]):>14}" for s in SPLITS
        )
        total = "-" if row.get("total") is None else row["total"]
        prev = row.get("percent_of_previous")
        orig = row.get("percent_of_original")
        print(f"{row['stage']:<44}{cells}{total:>9}"
              f"{('-' if prev is None else f'{prev}%'):>9}"
              f"{('-' if orig is None else f'{orig}%'):>9}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("STAGE 4 (v2): MERGING SPLITS")
    print("=" * 70)

    positives, negatives, pairs = [], [], []
    per_split = {}

    for split in SPLITS:
        positive = load_json(os.path.join(INPUT_DIR, f"positive_{split}.json"))
        negative = load_json(os.path.join(INPUT_DIR, f"negative_{split}.json"))
        pair = load_json(os.path.join(INPUT_DIR, f"validated_pairs_{split}.json"))

        if positive is None or negative is None or pair is None:
            print(f"  {split:<14} missing Stage 3 output - skipped")
            per_split[split] = {"status": "missing", "pairs": 0}
            continue

        status = "ok" if pair else "empty"
        print(f"  {split:<14} pairs: {len(pair):>5}   ({status})")
        per_split[split] = {"status": status, "pairs": len(pair)}

        positives.extend(positive)
        negatives.extend(negative)
        pairs.extend(pair)

    problems = check_integrity(positives, negatives, pairs)
    if problems:
        print("\nINTEGRITY CHECK FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)

    # Native negatives set aside by Stage 1
    natives, native_per_split = [], {}
    for split in SPLITS:
        rows = load_json(os.path.join(NATIVE_DIR, f"native_negative_{split}.json"))
        if rows is None:
            print(f"  {split:<14} native negatives: missing Stage 1 output - skipped")
            native_per_split[split] = None
            continue
        print(f"  {split:<14} native negatives: {len(rows):>5}")
        native_per_split[split] = len(rows)
        natives.extend(rows)

    native_ids = [r.get("id") for r in natives]
    clashes = set(native_ids) & {p.get("id") for p in pairs}
    repeated = [i for i, n in collections.Counter(native_ids).items() if n > 1]
    if clashes or repeated:
        print("\nINTEGRITY CHECK FAILED (native negatives):")
        if clashes:
            print(f"  - {len(clashes)} ids also used by pairs, e.g. {sorted(clashes)[:3]}")
        if repeated:
            print(f"  - {len(repeated)} duplicate ids, e.g. {repeated[:3]}")
        raise SystemExit(1)

    save_json(positives, POSITIVE_OUTPUT)
    save_json(negatives, NEGATIVE_OUTPUT)
    save_json(pairs, PAIRS_OUTPUT)
    save_json(natives, NATIVE_OUTPUT)

    negation_types = collections.Counter(p.get("negation_type") for p in pairs)
    question_rules = collections.Counter(p.get("question_rule") for p in pairs)

    funnel = build_funnel({s: v["pairs"] for s, v in per_split.items()})

    report = {
        "per_split": per_split,
        "total_pairs": len(pairs),
        "native_negatives": {"per_split": native_per_split, "total": len(natives)},
        "negation_types": dict(negation_types),
        "question_rules": dict(question_rules),
        "funnel": funnel,
    }
    save_json(report, REPORT_FILE)

    print("\nIntegrity checks passed.")
    print(f"Total pairs: {len(pairs)}   native negatives set aside: {len(natives)}")
    print(f"\nSaved:\n  {POSITIVE_OUTPUT}\n  {NEGATIVE_OUTPUT}\n  {PAIRS_OUTPUT}"
          f"\n  {NATIVE_OUTPUT}\n  {REPORT_FILE}")

    print("\nPIPELINE FUNNEL SO FAR")
    print_funnel(funnel)


if __name__ == "__main__":
    main()