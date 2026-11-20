"""
Stage 5 (v2): finalize the dataset and split it into train / test.

This single stage replaces v1's Stages 5-9 and split_dataset.py:

  v1 05_fix_negation_placement.py   no longer needed: Stage 2 v2 never
  v1 06_rewrite_head_negated.py     negates JOIN keys and never produces
                                    head-attached questions
  v1 07_duplicate_QA.py             -> step 1 (exact duplicates) below;
                                       paraphrases are NOT removed
  v1 08_remove_empty_ans.py         -> step 2 (verification) below
  v1 09_extract_matching_positive.py-> positives and negatives are written
                                       together from the pairs file
  v1 split_dataset.py               -> step 3 (random split) below

Steps, applied to final_datasets/pairs_all.json from Stage 4:

  1. Exact duplicates      the same positive and negative question text
                           (Spider repeats a few questions verbatim).
  2. Verification          both SQL queries are executed again and must
                           return the stored answers; answers must be
                           non-empty and not NULL-only, and the negative
                           question must contain a negation word.
  3. Split                 the pairs are shuffled with RANDOM_SEED and the
                           first TRAIN_RATIO of them go to train, the rest
                           to test. Each pair is assigned on its own.

Paraphrases (Spider asks most SQL queries twice in different words, e.g.
"How many universities have a location that contains NY?" / "Find how many
school locations have the word 'NY'.") are NOT removed and NOT grouped: every
paraphrase stays in the dataset as an independent pair, and two paraphrases
of the same SQL may end up on different sides of the train/test split.

Native negatives (final_datasets/native_negatives_all.json from Stage 4) are
the natively negative Spider questions that Stage 1 set aside (Stage 2 does
not negate them again). Stage 1 also converted each one to a positive
partner. They are added after the generated pairs are finished, as full
pairs: the converted positive goes to the POSITIVE set and the original
native question to the NEGATIVE set, so both sets have the same size.

  N1. Exact duplicates     a native negative whose question text equals an
                           earlier native negative or a generated negative
                           question is dropped.
  N2. Verification         the SQL must still return the stored answer; the
                           question must contain a negation word and the
                           answer must not be empty / NULL-only (unless
                           ALLOW_EMPTY_ANSWERS). Questions Stage 1 marked
                           negation_mismatch="question_only" skip these two
                           checks: Stage 1 keeps all of them.
  N3. Positive partner     natives whose conversion failed in Stage 1
                           (conversion_failed) are dropped, unless
                           KEEP_UNCONVERTED_NATIVES is True (then they are
                           negative-only and the two sets differ in size).
                           The positive SQL is executed again and must
                           return the stored positive answer.
  N4. Split                shuffled with RANDOM_SEED and cut at TRAIN_RATIO,
                           separately from the pairs, so both kinds of
                           pair are split 70/30. A native's positive and
                           negative are always in the same split.

Every record carries `source_type`. Negative records: "generated" (made by
Stage 2 from a positive) or "native_negative" (written by Spider's
annotators). Positive records: "original_positive" (Spider question) or
"converted_from_negative" (positive partner of a native negative).

Outputs (final_datasets/), with v1's names so the annotation script works:

  positive_all_ultimate_final.json   negative_all_ultimate_final.json
  pos_train.json  pos_test.json      neg_train.json  neg_test.json
                                     (both = generated pairs + native pairs)
  pairs_final.csv                    one row per pair, for annotation
  dropped_final.json                 every pair / native negative removed
                                     here, with reason
  stage5_report.json                 counts and the complete funnel
"""

import os
import re
import csv
import json
import time
import random
import sqlite3
import collections


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "datasets/pairs_all.json"
NATIVE_FILE = "datasets/native_negatives_all.json"

# Natives whose positive conversion failed in Stage 1:
#   False -> dropped, so positive and negative sets have the same size
#   True  -> kept as negative-only records (sets differ in size)
KEEP_UNCONVERTED_NATIVES = False
STAGE4_REPORT = "datasets/stage4_report.json"
DATABASE_DIR = "data/raw/database"
OUTPUT_DIR = "final_datasets"

VERIFY_BY_EXECUTION = True
# Keep answers that are empty or only NULLs. Keep equal to Stages 3 and 4.
# (Native negatives marked negation_mismatch="question_only" are kept even
# when their answer is empty: Stage 1 keeps all of them.)
ALLOW_EMPTY_ANSWERS = False
TRAIN_RATIO = 0.7
RANDOM_SEED = 42
QUERY_TIMEOUT_SECONDS = 60

SPLITS = ["train_spider", "train_others", "dev"]

NEGATION_WORDS = re.compile(
    r"\b(?:not|no|never|without|none|neither|nor|cannot)\b|n't\b", re.I
)

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


def normalize_text(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def normalize_sql(sql):
    sql = re.sub(r"\s+", " ", (sql or "").strip().rstrip(";"))
    sql = re.sub(r"\"([^\"]*)\"", r"'\1'", sql)
    return sql.lower()


def is_empty_or_null(answer):
    if not answer:
        return True
    return all(v is None or str(v).strip() == "" for row in answer for v in row)


def find_database(db_id):
    folder = os.path.join(DATABASE_DIR, db_id)
    standard = os.path.join(folder, f"{db_id}.sqlite")
    if os.path.exists(standard):
        return standard
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            if filename.endswith(".sqlite"):
                return os.path.join(folder, filename)
    return None


def execute_query(db_path, sql):
    connection = None
    try:
        connection = sqlite3.connect(db_path)
        connection.text_factory = lambda b: b.decode("utf-8", "ignore")
        started = time.time()
        connection.set_progress_handler(
            lambda: 1 if time.time() - started > QUERY_TIMEOUT_SECONDS else 0, 10000
        )
        rows = connection.execute(sql).fetchall()
        return True, [[None if v is None else str(v) for v in row] for row in rows]
    except Exception:
        return False, None
    finally:
        if connection is not None:
            connection.close()


def same_answer(stored, fresh):
    def norm(answer):
        return sorted(tuple("" if v is None else str(v) for v in row) for row in answer)
    return norm(stored) == norm(fresh)


# ============================================================
# STEPS
# ============================================================

def remove_exact_duplicates(pairs, dropped):
    kept, seen = [], set()
    for pair in pairs:
        key = (normalize_text(pair["positive_question"]), normalize_text(pair["negative_question"]))
        if key in seen:
            dropped.append(dict(pair, drop_reason="exact_duplicate"))
            continue
        seen.add(key)
        kept.append(pair)
    return kept


def verify(pairs, dropped, reasons):
    kept = []
    for pair in pairs:
        reason = None
        if not ALLOW_EMPTY_ANSWERS and is_empty_or_null(pair["positive_answer"]):
            reason = "positive_answer_empty_or_null"
        elif not ALLOW_EMPTY_ANSWERS and is_empty_or_null(pair["negative_answer"]):
            reason = "negative_answer_empty_or_null"
        elif len(NEGATION_WORDS.findall(pair["negative_question"])) <= \
                len(NEGATION_WORDS.findall(pair["positive_question"])):
            reason = "negative_question_without_added_negation"
        elif VERIFY_BY_EXECUTION:
            db_path = find_database(pair["db_id"])
            if db_path is None:
                reason = "database_missing"
            else:
                ok_p, fresh_p = execute_query(db_path, pair["positive_sql"])
                ok_n, fresh_n = execute_query(db_path, pair["negative_sql"])
                if not (ok_p and ok_n):
                    reason = "execution_failed"
                elif not same_answer(pair["positive_answer"], fresh_p):
                    reason = "positive_answer_changed"
                elif not same_answer(pair["negative_answer"], fresh_n):
                    reason = "negative_answer_changed"
        if reason:
            reasons[reason] += 1
            dropped.append(dict(pair, drop_reason=reason))
            continue
        kept.append(pair)
    return kept


def remove_native_duplicates(natives, final_pairs, dropped):
    """Drop a native negative whose question text was already seen, either
    as a generated negative question or as an earlier native negative."""
    seen = {normalize_text(p["negative_question"]) for p in final_pairs}
    kept = []
    for rec in natives:
        key = normalize_text(rec["question"])
        if key in seen:
            dropped.append(dict(rec, drop_reason="exact_duplicate"))
            continue
        seen.add(key)
        kept.append(rec)
    return kept


def verify_native(natives, dropped, reasons):
    kept = []
    for rec in natives:
        reason = None
        # Stage 1 already found negation in "question_only" questions (with a
        # wider word list: also non, non-, outside) and keeps them whatever
        # their answer, so the answer and negation-word checks skip them.
        question_only = rec.get("negation_mismatch") == "question_only"
        if not ALLOW_EMPTY_ANSWERS and not question_only and is_empty_or_null(rec["answer"]):
            reason = "native_answer_empty_or_null"
        elif rec.get("negation_mismatch") not in ("sql_only", "question_only") and \
                not NEGATION_WORDS.search(rec["question"]):
            reason = "native_question_without_negation_word"
        elif VERIFY_BY_EXECUTION:
            db_path = find_database(rec["db_id"])
            if db_path is None:
                reason = "database_missing"
            else:
                ok, fresh = execute_query(db_path, rec["sql"])
                if not ok:
                    reason = "native_execution_failed"
                elif not same_answer(rec["answer"], fresh):
                    reason = "native_answer_changed"
        if reason:
            reasons[reason] += 1
            dropped.append(dict(rec, drop_reason=reason))
            continue
        kept.append(rec)
    return kept


def keep_converted_natives(natives, dropped, reasons):
    """N3: keep natives that have a positive partner (or all of them when
    KEEP_UNCONVERTED_NATIVES), and re-check the positive answer."""
    kept = []
    for rec in natives:
        reason = None
        if not rec.get("positive_question"):
            if not KEEP_UNCONVERTED_NATIVES:
                reason = f"native_not_converted:{rec.get('conversion_failed') or 'unknown'}"
        elif VERIFY_BY_EXECUTION:
            db_path = find_database(rec["db_id"])
            if db_path is None:
                reason = "database_missing"
            else:
                ok, fresh = execute_query(db_path, rec["positive_sql"])
                if not ok:
                    reason = "native_positive_execution_failed"
                elif not same_answer(rec["positive_answer"], fresh):
                    reason = "native_positive_answer_changed"
        if reason:
            reasons[reason] += 1
            dropped.append(dict(rec, drop_reason=reason))
            continue
        kept.append(rec)
    return kept


def random_split(pairs):
    """Shuffle the pairs and cut them into train and test.
    Every pair is assigned on its own (no paraphrase grouping)."""
    shuffled = list(pairs)
    random.Random(RANDOM_SEED).shuffle(shuffled)
    target = round(TRAIN_RATIO * len(shuffled))
    return shuffled[:target], shuffled[target:]


def to_positive(pair, split):
    return {
        "id": pair["id"],
        "source_dataset": pair["source_dataset"],
        "db_id": pair["db_id"],
        "split": split,
        "source_type": "original_positive",
        "question": pair["positive_question"],
        "sql": pair["positive_sql"],
        "answer": pair["positive_answer"],
        "answer_count": len(pair["positive_answer"]),
    }


def to_negative(pair, split):
    return {
        "id": pair["id"],
        "source_dataset": pair["source_dataset"],
        "db_id": pair["db_id"],
        "split": split,
        "source_type": "generated",
        "question": pair["negative_question"],
        "sql": pair["negative_sql"],
        "answer": pair["negative_answer"],
        "answer_count": len(pair["negative_answer"]),
        "negation_type": pair.get("negation_type"),
        "question_rule": pair.get("question_rule"),
    }


def native_to_positive(rec, split):
    return {
        "id": rec["id"],
        "source_dataset": rec["source_dataset"],
        "db_id": rec["db_id"],
        "split": split,
        "source_type": "converted_from_negative",
        "question": rec["positive_question"],
        "sql": rec["positive_sql"],
        "answer": rec["positive_answer"],
        "answer_count": len(rec["positive_answer"]),
        "conversion": rec.get("conversion"),
    }


def native_positives(recs, split_of=None, split=None):
    """Positive records of the natives that have one."""
    return [native_to_positive(r, split or split_of[id(r)])
            for r in recs if r.get("positive_question")]


def native_to_negative(rec, split):
    return {
        "id": rec["id"],
        "source_dataset": rec["source_dataset"],
        "db_id": rec["db_id"],
        "split": split,
        "source_type": "native_negative",
        "question": rec["question"],
        "sql": rec["sql"],
        "answer": rec["answer"],
        "answer_count": len(rec["answer"]),
        "negation_type": None,
        "question_rule": None,
        "sql_negation": rec.get("sql_negation"),
        "negation_mismatch": rec.get("negation_mismatch"),
    }


def write_csv(rows, native_rows, path):
    fields = ["id", "split", "db_id", "source_dataset", "source_type", "negation_mismatch",
              "negation_type",
              "question_rule", "positive_question", "negative_question", "positive_sql",
              "negative_sql", "positive_answer", "negative_answer"]
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split, pair in rows:
            writer.writerow({
                "id": pair["id"], "split": split, "db_id": pair["db_id"],
                "source_dataset": pair["source_dataset"],
                "source_type": "generated",
                "negation_mismatch": "",
                "negation_type": pair.get("negation_type"),
                "question_rule": pair.get("question_rule"),
                "positive_question": pair["positive_question"],
                "negative_question": pair["negative_question"],
                "positive_sql": pair["positive_sql"],
                "negative_sql": pair["negative_sql"],
                "positive_answer": json.dumps(pair["positive_answer"], ensure_ascii=False),
                "negative_answer": json.dumps(pair["negative_answer"], ensure_ascii=False),
            })
        for split, rec in native_rows:
            writer.writerow({
                "id": rec["id"], "split": split, "db_id": rec["db_id"],
                "source_dataset": rec["source_dataset"],
                "source_type": "native_negative",
                "negation_mismatch": rec.get("negation_mismatch") or "",
                "negation_type": "", "question_rule": "",
                "positive_question": rec.get("positive_question") or "",
                "positive_sql": rec.get("positive_sql") or "",
                "positive_answer": (json.dumps(rec["positive_answer"], ensure_ascii=False)
                                    if rec.get("positive_answer") is not None else ""),
                "negative_question": rec["question"],
                "negative_sql": rec["sql"],
                "negative_answer": json.dumps(rec["answer"], ensure_ascii=False),
            })


def per_split(pairs):
    counts = collections.Counter(p["source_dataset"] for p in pairs)
    return {s: counts.get(s, 0) for s in SPLITS}


def funnel_row(label, counts, previous_total, original_total):
    total = sum(counts.values())
    return {
        "stage": label,
        "counts": counts,
        "total": total,
        "percent_of_previous": round(100 * total / previous_total, 1) if previous_total else None,
        "percent_of_original": round(100 * total / original_total, 1) if original_total else None,
    }


def print_funnel(rows):
    header = f"{'Stage':<46}" + "".join(f"{s:>14}" for s in SPLITS) + \
        f"{'Total':>13}{'% prev':>9}{'% orig':>9}"
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = "".join(
            f"{('-' if row['counts'].get(s) is None else row['counts'][s]):>14}" for s in SPLITS
        )
        prev, orig = row.get("percent_of_previous"), row.get("percent_of_original")
        print(f"{row['stage']:<46}{cells}{row['total']:>13}"
              f"{('-' if prev is None else f'{prev}%'):>9}"
              f"{('-' if orig is None else f'{orig}%'):>9}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("STAGE 5 (v2): FINALIZE AND SPLIT")
    print(f"Paraphrases: kept, not grouped   verify by execution: {VERIFY_BY_EXECUTION}   "
          f"train ratio: {TRAIN_RATIO}   seed: {RANDOM_SEED}")
    print("=" * 70)

    pairs = load_json(INPUT_FILE)
    if pairs is None:
        raise SystemExit(f"ERROR: {INPUT_FILE} not found - run Stage 4 first.")
    print(f"Loaded {len(pairs)} pairs")

    dropped = []
    verification_reasons = collections.Counter()

    after_exact = remove_exact_duplicates(pairs, dropped)
    final = verify(after_exact, dropped, verification_reasons)
    train, test = random_split(final)

    # Native negatives are added after the generated pairs are finished.
    natives = load_json(NATIVE_FILE)
    if natives is None:
        print(f"WARNING: {NATIVE_FILE} not found - no native negatives added.")
        natives = []
    print(f"Loaded {len(natives)} native negatives")
    native_reasons = collections.Counter()
    native_after_dup = remove_native_duplicates(natives, final, dropped)
    native_verified = verify_native(native_after_dup, dropped, native_reasons)
    native_final = keep_converted_natives(native_verified, dropped, native_reasons)
    native_train, native_test = random_split(native_final)

    # --------------------------------------------------------
    # Write outputs
    # --------------------------------------------------------

    split_of = {id(p): "train" for p in train + native_train}
    split_of.update({id(p): "test" for p in test + native_test})

    def source_order(p):
        return (SPLITS.index(p["source_dataset"]) if p["source_dataset"] in SPLITS else 9, p["id"])

    ordered = sorted(final, key=source_order)
    native_ordered = sorted(native_final, key=source_order)

    save_json([to_positive(p, split_of[id(p)]) for p in ordered]
              + native_positives(native_ordered, split_of=split_of),
              os.path.join(OUTPUT_DIR, "positive_all_ultimate_final.json"))
    save_json([to_negative(p, split_of[id(p)]) for p in ordered]
              + [native_to_negative(r, split_of[id(r)]) for r in native_ordered],
              os.path.join(OUTPUT_DIR, "negative_all_ultimate_final.json"))
    save_json([to_positive(p, "train") for p in train] + native_positives(native_train, split="train"),
              os.path.join(OUTPUT_DIR, "pos_train.json"))
    save_json([to_negative(p, "train") for p in train]
              + [native_to_negative(r, "train") for r in native_train],
              os.path.join(OUTPUT_DIR, "neg_train.json"))
    save_json([to_positive(p, "test") for p in test] + native_positives(native_test, split="test"),
              os.path.join(OUTPUT_DIR, "pos_test.json"))
    save_json([to_negative(p, "test") for p in test]
              + [native_to_negative(r, "test") for r in native_test],
              os.path.join(OUTPUT_DIR, "neg_test.json"))
    save_json(dropped, os.path.join(OUTPUT_DIR, "dropped_final.json"))
    write_csv([(split_of[id(p)], p) for p in ordered],
              [(split_of[id(r)], r) for r in native_ordered],
              os.path.join(OUTPUT_DIR, "pairs_final.csv"))

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    funnel = (load_json(STAGE4_REPORT, {}) or {}).get("funnel", [])
    original_total = funnel[0]["total"] if funnel else None
    previous_total = funnel[-1]["total"] if funnel else len(pairs)

    for label, rows in (
        ("5a. Exact duplicates removed", after_exact),
        ("5b. Verified (final dataset)", final),
    ):
        row = funnel_row(label, per_split(rows), previous_total, original_total)
        funnel.append(row)
        previous_total = row["total"]

    # Native negatives have their own short funnel (they never were pairs).
    native_funnel = []
    native_previous = len(natives)
    for label, rows in (
        ("N0. Native negatives set aside in Stage 1", natives),
        ("N1. Exact duplicates removed", native_after_dup),
        ("N2. Verified", native_verified),
        ("N3. With positive partner (added)", native_final),
    ):
        row = funnel_row(label, per_split(rows), native_previous, original_total)
        native_funnel.append(row)
        native_previous = row["total"]

    report = {
        "settings": {"paraphrases": "kept, not grouped", "verify_by_execution": VERIFY_BY_EXECUTION,
                     "allow_empty_answers": ALLOW_EMPTY_ANSWERS,
                     "train_ratio": TRAIN_RATIO, "random_seed": RANDOM_SEED},
        "input_pairs": len(pairs),
        "dropped": dict(collections.Counter(d["drop_reason"] for d in dropped)),
        "verification_failures": dict(verification_reasons),
        "final_pairs": len(final),
        "train": {"pairs": len(train), "by_source": per_split(train)},
        "test": {"pairs": len(test), "by_source": per_split(test)},
        "native_negatives": {
            "input": len(natives),
            "final": len(native_final),
            "verification_failures": dict(native_reasons),
            "train": {"count": len(native_train), "by_source": per_split(native_train)},
            "test": {"count": len(native_test), "by_source": per_split(native_test)},
            "sql_negation": dict(collections.Counter(
                name for r in native_final for name in (r.get("sql_negation") or []))),
            "negation_mismatch": dict(collections.Counter(
                r.get("negation_mismatch") or "none" for r in native_final)),
        },
        "negative_totals": {"train": len(train) + len(native_train),
                            "test": len(test) + len(native_test),
                            "all": len(final) + len(native_final)},
        "positive_totals": {"train": len(train) + len(native_positives(native_train, split="train")),
                            "test": len(test) + len(native_positives(native_test, split="test")),
                            "all": len(final) + len(native_positives(native_final, split="all"))},
        "negation_types": dict(collections.Counter(p.get("negation_type") for p in final)),
        "question_rules": dict(collections.Counter(p.get("question_rule") for p in final)),
        "funnel": funnel,
        "native_funnel": native_funnel,
    }
    save_json(report, os.path.join(OUTPUT_DIR, "stage5_report.json"))

    print(f"\nDropped: {report['dropped'] or 'none'}")
    print(f"Final pairs: {len(final)}   train: {len(train)}   test: {len(test)}")
    print(f"Native pairs added: {len(native_final)}   train: {len(native_train)}   "
          f"test: {len(native_test)}")
    pos_tot, neg_tot = report["positive_totals"], report["negative_totals"]
    print(f"Positive data: {pos_tot['all']} (train {pos_tot['train']}, test {pos_tot['test']})   "
          f"Negative data: {neg_tot['all']} (train {neg_tot['train']}, test {neg_tot['test']})")
    print(f"\nOutputs written to {OUTPUT_DIR}/")
    print("\nCOMPLETE PIPELINE FUNNEL")
    print_funnel(funnel)
    print(f"{'Train / test split':<46}"
          + "".join(f"{str(per_split(train)[s]) + ' / ' + str(per_split(test)[s]):>14}" for s in SPLITS)
          + f"{str(len(train)) + ' / ' + str(len(test)):>13}")
    print("\nNATIVE NEGATIVES (added as pairs with their converted positive)")
    print_funnel(native_funnel)


if __name__ == "__main__":
    main()