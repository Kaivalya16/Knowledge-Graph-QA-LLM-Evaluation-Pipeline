"""
Stage 3 (v2): filter the negation candidates and keep ONE validated
positive/negative pair per source question.

Each candidate goes through the checks below, in order; the first failure
rejects it:

  positive_answer_empty          the positive answer has no rows
  negative_answer_empty          the negated SQL returns no rows
  negative_answer_null_only      it returns only NULLs (e.g. AVG over nothing)
  negative_answer_over_limit     it returns more than MAX_NEGATIVE_ROWS rows
  answers_identical              negating the condition changed nothing
  sql_identical / question_identical   sanity checks
  duplicate_pair                 exactly the same pair was already seen

Among the candidates of one question that pass, ONE is kept.

Changes from v1:

  1. Selection is no longer "first valid candidate in SQL order". v1's first
     candidate was usually a negated JOIN key, which is why Stage 5 had to
     repair or drop most pairs. Candidates are now ranked by how reliable
     their question rewrite is (RULE_PRIORITY), then by whether the
     predicate is in the main query rather than a subquery, then by SQL
     position, so the choice is deterministic.
  2. Answer identity is taken from Stage 2, which compared the FULL results
     (Stage 2 stores at most 100 rows per negated answer).
  3. NULL-only negative answers are rejected; v1 accepted [[None]].
  4. The row limit is a named setting (MAX_NEGATIVE_ROWS, default 10, as in
     v1), and an optional limit for positive answers is available.
  5. Every pair carries `question_rule` and `predicate`. Paraphrases (Spider
     asks most SQL queries twice in different words) are NOT grouped or
     labelled: every paraphrase is kept as an independent pair.
  6. Statistics are reported per candidate AND per question. A rejected
     question is counted under the reason of its candidate that got
     furthest through the checks. Everything goes to stage3_report.json.

Output files and their fields are a superset of v1's, so Stage 4 is unchanged.
"""

import os
import re
import json
import collections


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DATASETS = {
    "train_spider": "data/processed/negation_candidates/negated_candidates_train_spider.json",
    "train_others": "data/processed/negation_candidates/negated_candidates_train_others.json",
    "dev": "data/processed/negation_candidates/negated_candidates_dev.json",
}

CANONICAL_DIR = "data/processed/canonical_positive"
OUTPUT_DIR = "data/processed/final"
REPORT_FILE = os.path.join(OUTPUT_DIR, "stage3_report.json")

MAX_NEGATIVE_ROWS = 10
MAX_POSITIVE_ROWS = None   # e.g. 10 to cap positive answers too; None = no cap

# Lower number = preferred when one question has several valid candidates.
RULE_PRIORITY = {
    "copula": 0,           # "whose email is not X"
    "participle": 0,       # "the employee not named X"
    "between": 0,          # "price not between 60 and 120"
    "comparison": 1,       # "no more than 1 head"
    "like": 1,             # "does not contain 'Mary'"
    "relative_clause": 2,  # "who did not participate in any round"
    "preposition": 2,      # "not from France"
    "coordination": 2,     # "in London or not Glasgow" (last conjunct)
    "contrast": 3,         # "type code that is not 'CV'"
}

# Order of the checks; used to tell how far a candidate got.
CHECK_ORDER = [
    "positive_answer_empty",
    "positive_answer_over_limit",
    "negative_answer_empty",
    "negative_answer_null_only",
    "negative_answer_over_limit",
    "answers_identical",
    "sql_identical",
    "question_identical",
    "duplicate_pair",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def normalize_text(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def normalize_sql(sql):
    sql = re.sub(r"\s+", " ", (sql or "").strip().rstrip(";"))
    sql = re.sub(r"\"([^\"]*)\"", r"'\1'", sql)
    return sql.lower()


def normalize_answer(answer):
    return sorted(
        tuple("" if v is None else str(v).strip().lower() for v in row)
        if isinstance(row, list) else (str(row).strip().lower(),)
        for row in (answer or [])
    )


def null_only(answer):
    return bool(answer) and all(
        v is None for row in answer for v in (row if isinstance(row, list) else [row])
    )


def in_subquery(positive_sql, negated_sql):
    """True when the negation was applied inside parentheses (a subquery).
    The edit position is where the two SQL strings start to differ."""
    position = 0
    limit = min(len(positive_sql), len(negated_sql))
    while position < limit and positive_sql[position] == negated_sql[position]:
        position += 1
    prefix = positive_sql[:position]
    return prefix.count("(") > prefix.count(")")


def check_candidate(candidate):
    """Return the name of the first failed check, or None."""
    positive_answer = candidate.get("positive_answer") or []
    negative_answer = candidate.get("negated_answer") or []
    positive_count = candidate.get("positive_answer_count", len(positive_answer))
    negative_count = candidate.get("negated_answer_count", len(negative_answer))

    if positive_count == 0:
        return "positive_answer_empty"
    if MAX_POSITIVE_ROWS is not None and positive_count > MAX_POSITIVE_ROWS:
        return "positive_answer_over_limit"
    if negative_count == 0:
        return "negative_answer_empty"
    if null_only(negative_answer):
        return "negative_answer_null_only"
    if negative_count > MAX_NEGATIVE_ROWS:
        return "negative_answer_over_limit"

    identical = candidate.get("answers_identical")
    if identical is None:  # candidates from v1's Stage 2
        identical = normalize_answer(positive_answer) == normalize_answer(negative_answer)
    if identical:
        return "answers_identical"

    if normalize_sql(candidate.get("positive_sql")) == normalize_sql(candidate.get("negated_sql")):
        return "sql_identical"
    if normalize_text(candidate.get("positive_question")) == normalize_text(candidate.get("negated_question")):
        return "question_identical"
    return None


def selection_key(candidate):
    rule = candidate.get("question_rule")
    return (
        RULE_PRIORITY.get(rule, 9),
        1 if in_subquery(candidate.get("positive_sql", ""), candidate.get("negated_sql", "")) else 0,
        candidate.get("operator_position", 0),
    )


# ============================================================
# PROCESS ONE DATASET
# ============================================================

def process_dataset(dataset_name, input_file):
    print("\n" + "=" * 70)
    print(f"FILTERING AND VALIDATING: {dataset_name}")
    print("=" * 70)

    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        return None

    with open(input_file, "r", encoding="utf-8") as handle:
        candidates = json.load(handle)
    print(f"Loaded {len(candidates)} candidates")

    # Canonical positives give the number of questions that entered Stage 2.
    canonical_file = os.path.join(CANONICAL_DIR, f"canonical_positive_{dataset_name}.json")
    questions_in_stage2 = None
    if os.path.exists(canonical_file):
        with open(canonical_file, "r", encoding="utf-8") as handle:
            questions_in_stage2 = len(json.load(handle))

    by_source = collections.OrderedDict()
    for candidate in candidates:
        by_source.setdefault(candidate["source_id"], []).append(candidate)

    candidate_stats = collections.Counter()
    question_stats = collections.Counter()
    selected_rules = collections.Counter()
    selected_types = collections.Counter()

    accepted_pairs = []
    rejected = []
    seen_pairs = set()

    for source_id, group in by_source.items():
        valid = []
        furthest = None   # failed check reached by the best candidate

        for candidate in group:
            candidate_stats["input_candidates"] += 1
            reason = check_candidate(candidate)

            if reason is None:
                pair_key = (
                    normalize_text(candidate["positive_question"]),
                    normalize_sql(candidate["positive_sql"]),
                    normalize_text(candidate["negated_question"]),
                    normalize_sql(candidate["negated_sql"]),
                )
                if pair_key in seen_pairs:
                    reason = "duplicate_pair"
                else:
                    valid.append((candidate, pair_key))
                    continue

            candidate_stats[f"rejected:{reason}"] += 1
            rejected.append({
                "candidate_id": candidate.get("candidate_id"),
                "source_id": source_id,
                "reason": reason,
                "negated_question": candidate.get("negated_question"),
                "negated_sql": candidate.get("negated_sql"),
                "negated_answer_count": candidate.get("negated_answer_count"),
                "question_rule": candidate.get("question_rule"),
            })
            if furthest is None or CHECK_ORDER.index(reason) > CHECK_ORDER.index(furthest):
                furthest = reason

        question_stats["questions_with_candidates"] += 1

        if not valid:
            question_stats[f"rejected:{furthest}"] += 1
            continue

        valid.sort(key=lambda item: selection_key(item[0]))
        chosen, pair_key = valid[0]
        seen_pairs.add(pair_key)

        for other, _ in valid[1:]:
            candidate_stats["rejected:not_selected_same_source"] += 1
            rejected.append({
                "candidate_id": other.get("candidate_id"),
                "source_id": source_id,
                "reason": "not_selected_same_source",
                "negated_question": other.get("negated_question"),
                "question_rule": other.get("question_rule"),
            })

        candidate_stats["accepted"] += 1
        question_stats["accepted"] += 1
        selected_rules[chosen.get("question_rule")] += 1
        selected_types[chosen.get("negation_type")] += 1

        positive_answer = chosen["positive_answer"]
        negative_answer = chosen["negated_answer"]
        accepted_pairs.append({
            "id": source_id,
            "source_dataset": chosen.get("source_dataset"),
            "db_id": chosen.get("db_id"),
            "positive_question": chosen["positive_question"],
            "positive_sql": chosen["positive_sql"],
            "positive_answer": positive_answer,
            "negative_question": chosen["negated_question"],
            "negative_sql": chosen["negated_sql"],
            "negative_answer": negative_answer,
            "negation_type": chosen.get("negation_type"),
            "question_rule": chosen.get("question_rule"),
            "predicate": chosen.get("predicate"),
            "candidate_id": chosen.get("candidate_id"),
            "positive_answer_count": chosen.get("positive_answer_count", len(positive_answer)),
            "negative_answer_count": chosen.get("negated_answer_count", len(negative_answer)),
        })

    # --------------------------------------------------------
    # Split pairs into the positive and negative datasets
    # --------------------------------------------------------

    positive_dataset = []
    negative_dataset = []
    for pair in accepted_pairs:
        shared = {
            "id": pair["id"],
            "source_dataset": pair["source_dataset"],
            "db_id": pair["db_id"],
        }
        positive_dataset.append(dict(
            shared,
            question=pair["positive_question"],
            sql=pair["positive_sql"],
            answer=pair["positive_answer"],
            answer_count=pair["positive_answer_count"],
        ))
        negative_dataset.append(dict(
            shared,
            question=pair["negative_question"],
            sql=pair["negative_sql"],
            answer=pair["negative_answer"],
            answer_count=pair["negative_answer_count"],
            negation_type=pair["negation_type"],
            question_rule=pair["question_rule"],
        ))

    outputs = {
        f"validated_pairs_{dataset_name}.json": accepted_pairs,
        f"positive_{dataset_name}.json": positive_dataset,
        f"negative_{dataset_name}.json": negative_dataset,
        f"rejected_candidates_{dataset_name}.json": rejected,
    }
    for filename, content in outputs.items():
        with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8") as handle:
            json.dump(content, handle, indent=2, ensure_ascii=False)

    if questions_in_stage2 is not None:
        question_stats["questions_in_stage2"] = questions_in_stage2
        question_stats["questions_without_candidates"] = (
            questions_in_stage2 - question_stats["questions_with_candidates"]
        )

    print(f"\nRESULTS: {dataset_name}")
    print("-" * 70)
    for key in sorted(question_stats):
        print(f"  [question]  {key:<44} {question_stats[key]:>6}")
    for key in sorted(candidate_stats):
        print(f"  [candidate] {key:<44} {candidate_stats[key]:>6}")
    print(f"\nSaved to: {OUTPUT_DIR}")

    return {
        "questions": dict(question_stats),
        "candidates": dict(candidate_stats),
        "selected_rules": dict(selected_rules),
        "selected_negation_types": dict(selected_types),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("STAGE 3 (v2): FILTERING AND VALIDATING NEGATED QA PAIRS")
    print(f"Negative answers: 1..{MAX_NEGATIVE_ROWS} rows; positive cap: {MAX_POSITIVE_ROWS}")
    print("=" * 70)

    report = {"settings": {"max_negative_rows": MAX_NEGATIVE_ROWS,
                           "max_positive_rows": MAX_POSITIVE_ROWS,
                           "rule_priority": RULE_PRIORITY}}
    for dataset_name, input_file in INPUT_DATASETS.items():
        result = process_dataset(dataset_name, input_file)
        if result is not None:
            report[dataset_name] = result

    splits = [k for k in report if k != "settings"]
    total_candidates = sum(report[k]["candidates"].get("input_candidates", 0) for k in splits)
    total_questions = sum(report[k]["questions"].get("questions_with_candidates", 0) for k in splits)
    total_accepted = sum(report[k]["questions"].get("accepted", 0) for k in splits)
    report["_summary"] = {
        "input_candidates": total_candidates,
        "questions_with_candidates": total_questions,
        "accepted_pairs": total_accepted,
        "accepted_percent_of_questions":
            round(100 * total_accepted / total_questions, 2) if total_questions else 0,
    }
    with open(REPORT_FILE, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\n" + "=" * 70)
    print(f"CANDIDATES: {total_candidates}   QUESTIONS: {total_questions}   "
          f"ACCEPTED PAIRS: {total_accepted} "
          f"({report['_summary']['accepted_percent_of_questions']}%)")
    print(f"Report: {REPORT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()