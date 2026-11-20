#!/usr/bin/env python3
"""
filter_valid_pairs.py
---------------------
Keep only valid positive/negative QA pairs. A pair (same id in both files) is
kept only if BOTH sides pass every check; otherwise both sides are removed, so
the two output files always contain the same ids.

Paraphrases are kept. Items that share one gold SQL (Spider's paraphrased
questions) are judged together for SQL-level problems, so a paraphrase is never
kept with SQL that was rejected for its twin.

The row cap applies to the NEGATIVE answer only (--neg-row-cap, default 10).
Positive answers are never capped.

Checks (reason codes in dropped.csv)
  UNPAIRED                id missing from one file, or db_id differs
  SCOPE_ERROR:<rule>      negative SQL has the wrong negation scope
                          (INTERSECT_AND_NOT, INTERSECT_VERB_NEG, JOIN_SCOPE, ZERO_COUNT)
  NEGATION_UNSOUND:<rule> ZERO_COUNT without a safe rewrite, INTERSECT_DEGENERATE,
                          OR_NOT_VACUOUS, INTERSECT_SCOPE_CONFLICT, SETOP_UNALIGNED
  AMBIGUOUS:<rule>        JOIN_SCOPE_AMBIGUOUS, AND_OR_PRECEDENCE   (kept with --keep-ambiguous)
  COMPARATOR_MISMATCH     negative question and SQL disagree on > / <
  POSITIVE_HAS_NEGATION   positive question or SQL is itself negated (EXCEPT, NOT IN, !=, ...)
  NEGATIVE_NO_NEGATION    negative question has no negation word, or SQL equals the positive SQL
  SAME_ANSWER             negative and positive answers have the same distinct rows
  EMPTY_ANSWER            either answer is empty or contains only NULL
  NEG_OVER_CAP            negative answer has more rows than --neg-row-cap
  DUPLICATE_QUESTION      same database, same negative and same positive question as an earlier pair
  SQL_ERROR               (with --db-dir) a query fails on the database

With --db-dir every query is re-executed and the stored answers are refreshed.
With --db-dir --fix, SCOPE_ERROR items are repaired (same rewrites as
fix_negative_sql.py) instead of dropped, as long as the repaired query runs,
returns rows, differs from the positive answer and respects the negative cap.

Usage
  python filter_valid_pairs.py --neg negative.json --pos positive.json \
      --tables tables.json --out filtered/ [--db-dir spider/database [--fix]]

Needs fix_negative_sql.py in the same folder.
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fix_negative_sql as F  # noqa: E402

SCOPE_RULES = {"INTERSECT_AND_NOT", "INTERSECT_VERB_NEG", "JOIN_SCOPE", "ZERO_COUNT"}
UNSOUND_RULES = {"INTERSECT_DEGENERATE", "OR_NOT_VACUOUS", "INTERSECT_SCOPE_CONFLICT", "SETOP_UNALIGNED",
                 "ZERO_COUNT"}
AMBIGUOUS_RULES = {"JOIN_SCOPE_AMBIGUOUS", "AND_OR_PRECEDENCE"}

NEG_WORD = re.compile(r"\b(?:not|no|never|none|without|except|excluding|neither|nor|cannot|other than|non)\b|n't",
                      re.I)
NEG_SQL = re.compile(r"\bexcept\b|\bnot\s+in\b|\bnot\s+like\b|\bnot\s+exists\b|\bnot\s+between\b|\bnot\s*\(|!=|<>",
                     re.I)
# quoted literals only; an apostrophe inside a word (head's, haven't) does not open a quote
QUOTED = re.compile(r"(?<![A-Za-z])'[^']*'(?![A-Za-z])|\"[^\"]*\"|\u2018[^\u2019]*\u2019|\u201c[^\u201d]*\u201d")


def norm_text(s):
    return re.sub(r"\s+", " ", s.lower().replace('"', "'")).strip().rstrip(";").strip()


def has_negation_word(question):
    return bool(NEG_WORD.search(QUOTED.sub(" ", question)))


def distinct(answer):
    return {json.dumps([None if v is None else str(v).strip().lower() for v in row]) for row in answer}


def all_null(answer):
    return bool(answer) and all(v is None for row in answer for v in row)


def audit_negative(rec, pos, schema):
    """Findings from the fix_negative_sql rules for one negative item."""
    cue = F.cues(rec["question"])
    findings = []
    try:
        findings += F.rule_intersect(rec, pos, cue)
        if not findings:
            findings += F.rule_join_scope(rec, pos, cue, schema)
            findings += F.rule_zero_count(rec, pos, cue, schema)
        findings += F.rule_or_not(rec, pos, cue)
    except Exception as e:  # parser surprise: do not silently pass the item
        findings.append(F.Finding("PARSER_ERROR", "review", repr(e)))
    return findings


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--neg", required=True)
    ap.add_argument("--pos", required=True)
    ap.add_argument("--tables", required=True, help="Spider tables.json")
    ap.add_argument("--out", default="filtered")
    ap.add_argument("--db-dir", default=None, help="<db-dir>/<db_id>/<db_id>.sqlite; re-executes every query")
    ap.add_argument("--fix", action="store_true", help="repair SCOPE_ERROR items instead of dropping (needs --db-dir)")
    ap.add_argument("--neg-row-cap", type=int, default=10, help="max rows in a NEGATIVE answer (0 = no cap)")
    ap.add_argument("--keep-ambiguous", action="store_true", help="keep JOIN_SCOPE_AMBIGUOUS / AND_OR_PRECEDENCE items")
    args = ap.parse_args()
    if args.fix and not args.db_dir:
        sys.exit("--fix needs --db-dir (repaired queries must be executed to get their answers)")

    neg = json.load(open(args.neg))
    pos = json.load(open(args.pos))
    schemas = {d["db_id"]: F.Schema(d) for d in json.load(open(args.tables))}
    ex = F.Executor(args.db_dir, schemas)
    os.makedirs(args.out, exist_ok=True)
    pos_by_id = {r["id"]: r for r in pos}
    neg_by_id = {r["id"]: r for r in neg}

    reasons = defaultdict(list)      # id -> [reason]
    fixes = {}                       # id -> (rule, new_sql)
    notes = defaultdict(list)        # id -> informational notes

    # ---- pairing
    for i in set(neg_by_id) ^ set(pos_by_id):
        reasons[i].append("UNPAIRED")
    paired = [r for r in neg if r["id"] in pos_by_id]
    for r in paired:
        if r["db_id"] != pos_by_id[r["id"]]["db_id"]:
            reasons[r["id"]].append("UNPAIRED")

    # ---- per-item checks
    sql_group = defaultdict(list)    # (db, negative sql) -> ids   (paraphrases share SQL)
    for r in paired:
        sql_group[(r["db_id"], norm_text(r["sql"]))].append(r["id"])

    group_reason, group_fix = {}, {}
    for r in paired:
        i, p = r["id"], pos_by_id[r["id"]]
        schema = schemas.get(r["db_id"])
        if schema is None:
            reasons[i].append("UNPAIRED")
            continue
        key = (r["db_id"], norm_text(r["sql"]))

        # negative SQL / question semantics
        for f in audit_negative(r, p, schema):
            if f.rule in SCOPE_RULES and f.level in ("auto", "medium") and f.new_sql:
                group_fix.setdefault(key, (f.rule, f.new_sql))
            elif f.rule in UNSOUND_RULES or f.rule in SCOPE_RULES or f.rule == "PARSER_ERROR":
                group_reason.setdefault(key, f"NEGATION_UNSOUND:{f.rule}")
            elif f.rule in AMBIGUOUS_RULES:
                if not args.keep_ambiguous:
                    reasons[i].append(f"AMBIGUOUS:{f.rule}")
                else:
                    notes[i].append(f.rule)
        if F.comparator_mismatches(r["question"], r["sql"]):
            reasons[i].append("COMPARATOR_MISMATCH")
        q2 = F.fix_gloss(r["question"])
        if q2 != r["question"]:
            r["question"] = q2
            notes[i].append("question gloss corrected")

        # polarity of each side
        if has_negation_word(p["question"]) or NEG_SQL.search(p["sql"]):
            reasons[i].append("POSITIVE_HAS_NEGATION")
        if not has_negation_word(r["question"]) or norm_text(r["sql"]) == norm_text(p["sql"]) \
                or norm_text(r["question"]) == norm_text(p["question"]):
            reasons[i].append("NEGATIVE_NO_NEGATION")

    # ---- SQL-level verdicts apply to every paraphrase that shares the SQL
    for key, ids in sql_group.items():
        if key in group_reason:
            for i in ids:
                reasons[i].append(group_reason[key])
        elif key in group_fix:
            rule, new_sql = group_fix[key]
            for i in ids:
                if args.fix:
                    fixes[i] = (rule, new_sql)
                else:
                    reasons[i].append(f"SCOPE_ERROR:{rule}")

    # ---- execution: repair, refresh, verify
    refreshed = 0
    if ex.real:
        for r in paired:
            i, p = r["id"], pos_by_id[r["id"]]
            if i in fixes:
                rule, new_sql = fixes[i]
                rows, err = ex.run(r["db_id"], new_sql)
                if err or not rows:
                    reasons[i].append(f"SCOPE_ERROR:{rule}")
                    notes[i].append("repair failed: " + (err or "no rows"))
                else:
                    r["sql"] = new_sql
                    notes[i].append(f"repaired ({rule})")
            for side, item in (("negative", r), ("positive", p)):
                rows, err = ex.run(item["db_id"], item["sql"])
                if err:
                    reasons[i].append("SQL_ERROR")
                    notes[i].append(f"{side}: {err}")
                    continue
                ans = F.to_answer(rows)
                if not F.same_answer(ans, item["answer"]):
                    refreshed += 1
                    item["answer"], item["answer_count"] = ans, len(ans)

    # ---- answer checks (after any refresh)
    for r in paired:
        i, p = r["id"], pos_by_id[r["id"]]
        if not r["answer"] or not p["answer"] or all_null(r["answer"]) or all_null(p["answer"]):
            reasons[i].append("EMPTY_ANSWER")
        elif distinct(r["answer"]) == distinct(p["answer"]):
            reasons[i].append("SAME_ANSWER")
        if args.neg_row_cap and len(r["answer"]) > args.neg_row_cap:
            reasons[i].append("NEG_OVER_CAP")

    # ---- exact duplicates: same database, same negative AND same positive question.
    # Paraphrases (different wording on either side) are kept.
    seen = set()
    for r in paired:
        if reasons.get(r["id"]):
            continue
        k = (r["db_id"], norm_text(r["question"]), norm_text(pos_by_id[r["id"]]["question"]))
        if k in seen:
            reasons[r["id"]].append("DUPLICATE_QUESTION")
        seen.add(k)

    # ---- write
    drop = {i for i, rs in reasons.items() if rs}
    neg_out = [r for r in neg if r["id"] in pos_by_id and r["id"] not in drop]
    pos_out = [r for r in pos if r["id"] in neg_by_id and r["id"] not in drop]
    assert {r["id"] for r in neg_out} == {r["id"] for r in pos_out}

    json.dump(neg_out, open(os.path.join(args.out, "negative_valid.json"), "w"), indent=2, ensure_ascii=False)
    json.dump(pos_out, open(os.path.join(args.out, "positive_valid.json"), "w"), indent=2, ensure_ascii=False)
    with open(os.path.join(args.out, "dropped.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "split", "source_type", "reasons", "neg_question", "pos_question", "neg_sql", "pos_sql",
                    "neg_rows", "pos_rows", "notes"])
        for i in sorted(drop, key=lambda x: (sorted(set(reasons[x]))[0], x)):
            n, p = neg_by_id.get(i, {}), pos_by_id.get(i, {})
            w.writerow([i, n.get("split", p.get("split")), n.get("source_type", ""),
                        "; ".join(sorted(set(reasons[i]))), n.get("question", ""), p.get("question", ""),
                        n.get("sql", ""), p.get("sql", ""), n.get("answer_count", ""), p.get("answer_count", ""),
                        " | ".join(notes.get(i, []))])

    # paraphrase / leakage report on what is kept
    def shared_sql(items):
        train = {(r["db_id"], norm_text(r["sql"])) for r in items if r.get("split") == "train"}
        test = [r for r in items if r.get("split") == "test"]
        return sum((r["db_id"], norm_text(r["sql"])) in train for r in test), len(test)

    groups = Counter((r["db_id"], norm_text(r["sql"])) for r in pos_out)
    leak, n_test = shared_sql(neg_out)
    summary = {
        "pairs_before": len(paired), "pairs_kept": len(neg_out), "pairs_dropped": len(drop),
        "kept_by_split": dict(Counter(r.get("split") for r in neg_out)),
        "kept_by_source_type": dict(Counter(r.get("source_type") for r in neg_out)),
        "drop_reasons": dict(sorted(Counter(x for i in drop for x in set(reasons[i])).items())),
        "dropped_by_split": dict(Counter(neg_by_id.get(i, pos_by_id.get(i, {})).get("split") for i in drop)),
        "repaired_items": sum(1 for i in fixes if i not in drop),
        "answers_refreshed_from_db": refreshed,
        "executed_on_real_db": ex.real,
        "negative_row_cap": args.neg_row_cap,
        "max_rows_kept": {"negative": max(r["answer_count"] for r in neg_out),
                          "positive": max(r["answer_count"] for r in pos_out)},
        "paraphrases_kept": {"distinct_sql": len(groups), "items_in_paraphrase_groups": sum(v for v in groups.values() if v > 1)},
        "test_items_whose_sql_is_also_in_train": f"{leak} of {n_test}",
    }
    json.dump(summary, open(os.path.join(args.out, "filter_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))
    if not ex.real:
        print("\nNOTE: no --db-dir: scope errors were dropped, not repaired, and answers were not re-executed.")


if __name__ == "__main__":
    main()
