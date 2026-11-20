"""
Stage 1 (v2): build canonical POSITIVE QA examples from Spider and set the
natively NEGATIVE ones aside.

Every Spider question ends up in exactly one bucket:

  original_positive        the question carries no negation -> kept; these
                           are the input of Stage 2 (negation generation).
                           This includes questions whose SQL has a negation
                           operator the question does not show (e.g. "other
                           than", "except", a technical != ''), marked
                           negation_mismatch = "sql_only". Stage 2 never
                           flips an existing !=, NOT IN or NOT LIKE; it only
                           negates another predicate of the SQL.
  native_negative          question and SQL are both negative -> NOT negated
                           again by Stage 2; set aside unchanged in
                           native_negative_<split>.json. Each one is also
                           converted to a POSITIVE partner (question and SQL
                           de-negated, positive SQL executed), so Stage 5 can
                           add it as a pair: the positive goes to the
                           positive set, the original native negative to the
                           negative set, keeping the two sets the same size.
                           Natives whose conversion fails keep
                           `conversion_failed` = reason; Stage 5 drops them.
                           A question that is negative while its SQL has no
                           negation operator (the SQL expresses the negation
                           another way, e.g. count(*) = 0) is a native
                           negative too, marked negation_mismatch =
                           "question_only". These are ALWAYS kept: the
                           answer checks below do not apply to them.
  rejected                 with a single, recorded reason

Changes from v1 (each one recovers questions v1 discarded or fixes a bug):

  1. EXCEPT is recognised as SQL negation. Spider writes most "who have not ..."
     questions as `A EXCEPT B`; v1 missed them, so 224 negative questions were
     rejected as "question negation without SQL negation".
  2. Quoted values are masked before detection, so 'No Response' or
     "Not Applicable" no longer make a positive question look negative.
  3. Comparatives such as "no more than", "no less than" and "not only ...
     but also" are not treated as negation (they map to <=, >=, INTERSECT).
  4. Column-to-column inequalities (`T1.title != T2.title`, a self-join
     device) are not treated as negation, so those questions stay positive.
  5. Natively negative questions are kept exactly as Spider wrote them, in
     a separate file, so that they are added after the generated negations
     (Stage 5). Their converted positive is stored next to them:
       SQL: every negated operator is flipped (!= -> =, NOT IN -> IN,
            NOT LIKE -> LIKE, NOT BETWEEN -> BETWEEN, IS NOT -> IS,
            NOT EXISTS -> EXISTS, a single top-level EXCEPT -> INTERSECT)
       question: grammar-aware de-negation ("does not have" -> "has",
            "did not attend" -> "attended", "without" -> "with", ...)
     The positive SQL must run, return a non-empty, non-NULL answer, and
     that answer must differ from the native negative answer.
     "question_only" natives have no negated SQL operator to flip, so their
     conversion fails (no_sql_negation_found).
  6. Native negatives must run and return a non-empty, non-NULL answer with
     at most NATIVE_MAX_ROWS rows (the same limit Stage 3 applies to the
     generated negatives).
  7. Questions with negation in the question but not in the SQL are no
     longer rejected; they are all set aside as native negatives, without
     the answer checks of point 6 (empty, NULL-only, row limit).
  8. Questions with negation in the SQL but not in the question are no
     longer rejected; they are kept as positives (same checks as any
     positive: non-empty, non-NULL answer).
  7. Databases with no rows (all six train_others databases in the Spider
     release) are rejected as `database_empty` instead of yielding [[0]]
     answers that can never form a valid pair later.
  8. Positive answers made only of NULLs are rejected.
  9. Queries run with a time limit and tolerant text decoding.
 10. A JSON report with per-reason counts is written for the statistics.

Outputs (data/processed/canonical_positive/):

  canonical_positive_<split>.json   positives, read by Stage 2 as before
  native_negative_<split>.json      natively negative questions, set aside
  rejected_<split>.json             every rejected question, with its reason
  stage1_report.json                counts per bucket and reason
"""

import os
import re
import json
import time
import sqlite3
import collections

from tqdm import tqdm


# ============================================================
# PATH CONFIGURATION
# ============================================================

DATASETS = {
    "train_spider": "data/raw/train_spider.json",
    "train_others": "data/raw/train_others.json",
    "dev": "data/raw/dev.json",
}

DATABASE_DIR = "data/raw/database"
OUTPUT_DIR = "data/processed/canonical_positive"
REPORT_FILE = os.path.join(OUTPUT_DIR, "stage1_report.json")

QUERY_TIMEOUT_SECONDS = 60

# Row limit for natively negative answers; keep equal to Stage 3's
# MAX_NEGATIVE_ROWS so native and generated negatives follow the same rule.
# None = no limit.
NATIVE_MAX_ROWS = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# MASKING
#
# Quoted values are replaced by placeholders so that negation words inside
# data values ('No Response', "Not Applicable") are neither detected nor
# rewritten. Contractions (don't) and possessives (students') are untouched
# because an opening quote must not follow a word character.
# ============================================================

QUESTION_QUOTE_RE = re.compile(
    r"\"[^\"]*\"|“[^”]*”|‘[^’]*’|(?<![\w'])'[^']+'(?![\w])"
)

# Phrases that contain "no"/"not" but are comparisons, not negations.
NON_NEGATING_PHRASES = re.compile(
    r"\bno\s+(?:more|less|fewer|greater|later|earlier|larger|smaller|"
    r"higher|lower|older|younger|longer|shorter|bigger)\s+than\b"
    r"|\bnot\s+only\b",
    re.IGNORECASE,
)


def mask_question(question):
    """Return (masked_text, restore_function)."""
    saved = []

    def stash(match):
        saved.append(match.group())
        return f"QVAL{len(saved) - 1}QVAL"

    masked = QUESTION_QUOTE_RE.sub(stash, question)
    masked = NON_NEGATING_PHRASES.sub(stash, masked)

    def restore(text):
        return re.sub(
            r"QVAL(\d+)QVAL",
            lambda m: saved[int(m.group(1))],
            text,
        )

    return masked, restore


def mask_sql_literals(sql):
    """Blank out string literals, keeping offsets identical."""
    return re.sub(
        r"'[^']*'|\"[^\"]*\"",
        lambda m: m.group()[0] + "x" * (len(m.group()) - 2) + m.group()[-1],
        sql,
    )


# ============================================================
# QUESTION NEGATION DETECTION
# ============================================================

QUESTION_NEGATION_RE = re.compile(
    r"\b(?:not|no|never|without|none|neither|nor|cannot|non)\b"
    r"|\bnon-"
    r"|n't\b"
    r"|\boutside\b",
    re.IGNORECASE,
)


def question_negation_cues(question):
    masked, _ = mask_question(question)
    return [m.group().lower() for m in QUESTION_NEGATION_RE.finditer(masked)]


def contains_question_negation(question):
    return bool(question_negation_cues(question))


# ============================================================
# SQL NEGATION DETECTION
#
# Each occurrence is returned as a span; the operator names are stored with
# the native negatives. Detection runs on the literal-masked SQL.
# ============================================================

IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\s*\.\s*[A-Za-z_][A-Za-z0-9_]*)?")


def rhs_is_column(sql, end):
    """True when the right-hand side of an operator is a bare column."""
    after = sql[end:].lstrip()
    if not after or after[0] in "'\"(" or re.match(r"-?\d", after):
        return False
    match = IDENTIFIER_RE.match(after)
    if not match:
        return False
    return match.group().lower() not in ("null", "true", "false")


def top_level_except_spans(masked_sql):
    """EXCEPT keywords outside any parentheses."""
    spans = []
    depth = 0
    for match in re.finditer(r"\(|\)|\bEXCEPT\b", masked_sql, re.IGNORECASE):
        token = match.group()
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
        elif depth == 0:
            spans.append(match.span())
    return spans


SQL_NEGATION_PATTERNS = [
    (r"\bNOT\s+EXISTS\b", "EXISTS", "not_exists_to_exists"),
    (r"\bNOT\s+IN\b", "IN", "not_in_to_in"),
    (r"\bNOT\s+LIKE\b", "LIKE", "not_like_to_like"),
    (r"\bNOT\s+BETWEEN\b", "BETWEEN", "not_between_to_between"),
    (r"\bIS\s+NOT\s+NULL\b", "IS NULL", "is_not_null_to_is_null"),
    (r"\bIS\s+NOT\b", "IS", "is_not_to_is"),
    (r"!=|<>", "=", "not_equal_to_equal"),
]


def find_sql_negations(sql):
    """
    Returns (operations, unsupported_reason).

    operations: list of dicts with start, end, replacement, name.
    unsupported_reason: set when the SQL is negative in an unusual way
    (nested EXCEPT, several EXCEPTs, NOT (...)). Such SQL still counts as
    negative; the reason is recorded with the native negative.
    """
    masked = mask_sql_literals(sql)
    operations = []
    occupied = []

    for pattern, replacement, name in SQL_NEGATION_PATTERNS:
        for match in re.finditer(pattern, masked, re.IGNORECASE):
            start, end = match.span()
            if any(start < o_end and end > o_start for o_start, o_end in occupied):
                continue
            # A self-join inequality is not a negation of the question.
            if name == "not_equal_to_equal" and rhs_is_column(masked, end):
                continue
            occupied.append((start, end))
            operations.append({
                "start": start,
                "end": end,
                "replacement": replacement,
                "name": name,
            })

    all_excepts = list(re.finditer(r"\bEXCEPT\b", masked, re.IGNORECASE))
    top_excepts = top_level_except_spans(masked)

    if all_excepts:
        if len(all_excepts) != len(top_excepts) or len(top_excepts) > 1:
            return operations, "unsupported_except_structure"
        start, end = top_excepts[0]
        operations.append({
            "start": start,
            "end": end,
            "replacement": "INTERSECT",
            "name": "except_to_intersect",
        })

    if re.search(r"\bNOT\s*\(", masked, re.IGNORECASE):
        return operations, "unsupported_not_parenthesis"

    operations.sort(key=lambda o: o["start"])
    return operations, None


def contains_sql_negation(sql):
    operations, unsupported = find_sql_negations(sql)
    return bool(operations) or unsupported is not None


def convert_sql_to_positive(sql):
    operations, unsupported = find_sql_negations(sql)
    if unsupported:
        return False, None, unsupported
    if not operations:
        return False, None, "no_sql_negation_found"
    converted = sql
    for operation in sorted(operations, key=lambda o: o["start"], reverse=True):
        converted = (
            converted[:operation["start"]]
            + operation["replacement"]
            + converted[operation["end"]:]
        )
    names = sorted({o["name"] for o in operations})
    return True, converted, "+".join(names)


# ============================================================
# QUESTION DE-NEGATION (grammar-aware)
# ============================================================

CONTRACTIONS = [
    (r"\bdoesn't\b", "does not"), (r"\bdon't\b", "do not"),
    (r"\bdidn't\b", "did not"), (r"\bisn't\b", "is not"),
    (r"\baren't\b", "are not"), (r"\bwasn't\b", "was not"),
    (r"\bweren't\b", "were not"), (r"\bhasn't\b", "has not"),
    (r"\bhaven't\b", "have not"), (r"\bhadn't\b", "had not"),
    (r"\bcan't\b", "can not"), (r"\bcannot\b", "can not"),
    (r"\bcouldn't\b", "could not"), (r"\bwon't\b", "will not"),
    (r"\bwouldn't\b", "would not"),
    # Spider typo: "teams that do no have ..." means "do not have"
    (r"\b(do|does|did)\s+no\b", r"\1 not"),
]

ADVERBS = {"yet", "ever", "currently", "actually", "even", "already", "also"}

IRREGULAR_PAST = {
    "have": "had", "do": "did", "go": "went", "take": "took", "make": "made",
    "win": "won", "get": "got", "give": "gave", "buy": "bought",
    "sell": "sold", "teach": "taught", "hold": "held", "pay": "paid",
    "send": "sent", "run": "ran", "see": "saw", "write": "wrote",
    "drive": "drove", "lead": "led", "spend": "spent", "keep": "kept",
    "leave": "left", "meet": "met", "begin": "began", "choose": "chose",
    "fly": "flew", "grow": "grew", "build": "built", "come": "came",
    "become": "became", "bring": "brought", "find": "found",
    "sing": "sang", "speak": "spoke", "steal": "stole", "wear": "wore",
    "plan": "planned", "ship": "shipped", "stop": "stopped",
    "prefer": "preferred", "occur": "occurred", "admit": "admitted",
    "enrol": "enrolled", "refer": "referred", "commit": "committed",
}

IRREGULAR_THIRD_PERSON = {"have": "has", "do": "does", "go": "goes", "be": "is"}


def third_person(verb):
    if verb in IRREGULAR_THIRD_PERSON:
        return IRREGULAR_THIRD_PERSON[verb]
    if re.search(r"[^aeiou]y$", verb):
        return verb[:-1] + "ies"
    if re.search(r"(s|x|z|ch|sh|o)$", verb):
        return verb + "es"
    return verb + "s"


def past_tense(verb):
    if verb in IRREGULAR_PAST:
        return IRREGULAR_PAST[verb]
    if verb.endswith("e"):
        return verb + "d"
    if re.search(r"[^aeiou]y$", verb):
        return verb[:-1] + "ied"
    return verb + "ed"


def keep_case(original, replacement):
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def resolve_do_support(text):
    """
    "does not have" -> "has", "do not own" -> "own",
    "did not attend" -> "attended". An adverb between "not" and the verb
    is kept in front of the verb ("does not yet have" -> "yet has").
    """
    pattern = re.compile(
        r"\b(do|does|did)\s+not\s+(?:(%s)\s+)?([A-Za-z]+)" % "|".join(ADVERBS),
        re.IGNORECASE,
    )

    def replace(match):
        auxiliary = match.group(1).lower()
        adverb = match.group(2)
        verb = match.group(3).lower()
        if auxiliary == "do":
            new_verb = verb
        elif auxiliary == "does":
            new_verb = third_person(verb)
        else:
            new_verb = past_tense(verb)
        words = ([adverb] if adverb else []) + [new_verb]
        return keep_case(match.group(1), " ".join(words))

    return pattern.sub(replace, text)


def convert_question_to_positive(question):
    masked, restore = mask_question(question)
    text = masked

    for pattern, replacement in CONTRACTIONS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    text = resolve_do_support(text)

    # Fixed phrases first.
    phrase_rules = [
        (r"\bwith\s+no\b", "with"),
        (r"\bwithout\b", "with"),
        (r"\boutside(\s+of)?\b", "in"),
        (r"\bnon-", ""),
        (r"\bnon\s+", ""),
        (r"\bnever\s+", ""),
        (r"\b(has|have|had)\s+no\b", r"\1"),
        # "by Statistics but not Psychology" (A EXCEPT B) -> "... and ..."
        (r"\bbut\s+not\b", "and"),
        (r"\bnot\s+", ""),
        # "cities that no customer lives in" -> "that some customer lives in"
        (r"\bno\s+(?=[a-z])", "some "),
        # "did not write any review" -> "wrote some review"
        (r"\bany\b", "some"),
    ]
    for pattern, replacement in phrase_rules:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([?.,;])", r"\1", text)

    if QUESTION_NEGATION_RE.search(text):
        return False, None, "unsupported_question_negation"

    positive = restore(text)
    if positive.strip().lower() == question.strip().lower():
        return False, None, "question_unchanged"
    return True, positive, "question_denegated"


# ============================================================
# DATABASE ACCESS
# ============================================================

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


_ROW_COUNT_CACHE = {}


def database_row_count(db_path):
    if db_path not in _ROW_COUNT_CACHE:
        connection = sqlite3.connect(db_path)
        try:
            tables = [
                row[0] for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            ]
            total = 0
            for table in tables:
                total += connection.execute(
                    f'SELECT count(*) FROM "{table}"'
                ).fetchone()[0]
        finally:
            connection.close()
        _ROW_COUNT_CACHE[db_path] = total
    return _ROW_COUNT_CACHE[db_path]


def execute_query(db_path, sql):
    connection = None
    try:
        connection = sqlite3.connect(db_path)
        connection.text_factory = lambda b: b.decode("utf-8", "ignore")
        started = time.time()
        connection.set_progress_handler(
            lambda: 1 if time.time() - started > QUERY_TIMEOUT_SECONDS else 0,
            10000,
        )
        rows = connection.execute(sql).fetchall()
        answer = [[None if v is None else str(v) for v in row] for row in rows]
        return True, answer, None
    except Exception as error:
        return False, None, str(error)
    finally:
        if connection is not None:
            connection.close()


def normalize_answer(answer):
    return sorted(
        tuple("" if v is None else str(v).strip().lower() for v in row)
        for row in answer
    )


def answer_is_null_only(answer):
    return bool(answer) and all(v is None for row in answer for v in row)


# ============================================================
# PROCESS ONE DATASET
# ============================================================

def convert_native_to_positive(question, sql, db_path, negative_answer):
    """Build the positive partner of a native negative.
    Returns (True, fields) or (False, reason)."""
    q_ok, positive_question, q_info = convert_question_to_positive(question)
    if not q_ok:
        return False, q_info
    s_ok, positive_sql, s_info = convert_sql_to_positive(sql)
    if not s_ok:
        return False, s_info
    ok, positive_answer, _error = execute_query(db_path, positive_sql)
    if not ok:
        return False, "converted_sql_execution_fail"
    if not positive_answer:
        return False, "empty_positive_answer"
    if answer_is_null_only(positive_answer):
        return False, "null_only_positive_answer"
    if normalize_answer(negative_answer) == normalize_answer(positive_answer):
        return False, "same_answer_after_conversion"
    return True, {
        "positive_question": positive_question,
        "positive_sql": positive_sql,
        "positive_answer": positive_answer,
        "conversion": {"question_conversion": q_info, "sql_conversion": s_info},
    }


def process_dataset(dataset_name, input_file):
    print("\n" + "=" * 70)
    print(f"PROCESSING DATASET: {dataset_name}")
    print("=" * 70)

    if not os.path.exists(input_file):
        print(f"ERROR: File not found: {input_file}")
        return None

    with open(input_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    print(f"Loaded {len(data)} examples")

    kept = []
    native_negatives = []
    rejected = []
    stats = collections.Counter()
    sql_negation_types = collections.Counter()

    def reject(example_id, reason, **extra):
        stats[f"rejected:{reason}"] += 1
        record = {"id": example_id, "source_dataset": dataset_name, "reason": reason}
        record.update(extra)
        rejected.append(record)

    for local_id, example in enumerate(tqdm(data, desc=dataset_name)):
        stats["total"] += 1
        example_id = f"{dataset_name}_{local_id}"

        question = example.get("question", "")
        sql = example.get("query", "")
        db_id = example.get("db_id", "")
        base = {"db_id": db_id, "original_question": question, "original_sql": sql}

        if not question or not sql or not db_id:
            reject(example_id, "missing_required_field", **base)
            continue

        db_path = find_database(db_id)
        if db_path is None:
            reject(example_id, "database_missing", **base)
            continue

        if database_row_count(db_path) == 0:
            reject(example_id, "database_empty", **base)
            continue

        ok, original_answer, error = execute_query(db_path, sql)
        if not ok:
            reject(example_id, "original_sql_execution_fail", error=error, **base)
            continue

        question_negative = contains_question_negation(question)
        sql_negative = contains_sql_negation(sql)

        # ---------------- Case 1 / 4: positive question ----------------
        # Case 4 (negation only in the SQL) is kept as a positive as well.
        if not question_negative:
            positive_mismatch = "sql_only" if sql_negative else None
            if not original_answer:
                reject(example_id, "empty_positive_answer", **base)
                continue
            if answer_is_null_only(original_answer):
                reject(example_id, "null_only_positive_answer", **base)
                continue
            kept.append({
                "id": example_id,
                "source_dataset": dataset_name,
                "db_id": db_id,
                "source_type": "original_positive",
                "original_question": question,
                "original_sql": sql,
                "original_answer": original_answer,
                "positive_question": question,
                "positive_sql": sql,
                "positive_answer": original_answer,
                "negation_mismatch": positive_mismatch,
            })
            stats["kept:original_positive"] += 1
            if positive_mismatch:
                stats["kept:original_positive_sql_only"] += 1
            continue

        # Case 3: negation only in the question -> native negative as well
        mismatch = "question_only" if not sql_negative else None

        # ---------------- Case 2 / 3: negative -> set aside ----------------
        # The answer checks apply to Case 2 only; Case 3 is always kept.
        if mismatch is None:
            if not original_answer:
                reject(example_id, "native_negative_empty_answer", **base)
                continue
            if answer_is_null_only(original_answer):
                reject(example_id, "native_negative_null_only_answer", **base)
                continue
            if NATIVE_MAX_ROWS is not None and len(original_answer) > NATIVE_MAX_ROWS:
                reject(example_id, "native_negative_over_limit",
                       answer_count=len(original_answer), **base)
                continue

        operations, unsupported = find_sql_negations(sql)
        sql_negation = sorted({o["name"] for o in operations})
        if unsupported:
            sql_negation.append(unsupported)
        record = {
            "id": example_id,
            "source_dataset": dataset_name,
            "db_id": db_id,
            "source_type": "native_negative",
            "question": question,
            "sql": sql,
            "answer": original_answer,
            "answer_count": len(original_answer),
            "question_negation_cues": question_negation_cues(question),
            "sql_negation": sql_negation,
            "negation_mismatch": mismatch,
        }

        # Positive partner (used by Stage 5 to add the native as a full pair)
        converted, result = convert_native_to_positive(question, sql, db_path, original_answer)
        if converted:
            record.update(result)
            record["conversion_failed"] = None
            stats["native_negative:converted_to_positive"] += 1
        else:
            record.update(positive_question=None, positive_sql=None,
                          positive_answer=None, conversion=None,
                          conversion_failed=result)
            stats[f"native_negative:not_converted:{result}"] += 1

        native_negatives.append(record)
        stats["set_aside:native_negative"] += 1
        if mismatch:
            stats[f"set_aside:native_negative_{mismatch}"] += 1
        for name in sql_negation:
            sql_negation_types[name] += 1

    stats["kept"] = len(kept)
    stats["native_negative"] = len(native_negatives)
    stats["rejected"] = len(rejected)

    output_file = os.path.join(OUTPUT_DIR, f"canonical_positive_{dataset_name}.json")
    native_file = os.path.join(OUTPUT_DIR, f"native_negative_{dataset_name}.json")
    rejected_file = os.path.join(OUTPUT_DIR, f"rejected_{dataset_name}.json")
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(kept, handle, indent=2, ensure_ascii=False)
    with open(native_file, "w", encoding="utf-8") as handle:
        json.dump(native_negatives, handle, indent=2, ensure_ascii=False)
    with open(rejected_file, "w", encoding="utf-8") as handle:
        json.dump(rejected, handle, indent=2, ensure_ascii=False)

    print(f"\nRESULTS: {dataset_name}")
    print("-" * 70)
    for key in sorted(stats):
        print(f"  {key:<50} {stats[key]:>6}")
    print(f"\nSaved: {output_file}\nSet aside: {native_file}\nRejected: {rejected_file}")

    return {"stats": dict(stats), "native_negative_sql_types": dict(sql_negation_types)}


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("STAGE 1 (v2): CANONICAL POSITIVES + NATIVE NEGATIVES SET ASIDE")
    print("=" * 70)

    report = {}
    for dataset_name, input_file in DATASETS.items():
        result = process_dataset(dataset_name, input_file)
        if result is not None:
            report[dataset_name] = result

    total_in = sum(r["stats"]["total"] for r in report.values())
    total_kept = sum(r["stats"]["kept"] for r in report.values())
    total_native = sum(r["stats"]["native_negative"] for r in report.values())
    report["_summary"] = {
        "input": total_in,
        "kept": total_kept,
        "native_negative_set_aside": total_native,
        "rejected": total_in - total_kept - total_native,
        "kept_percent": round(100 * total_kept / total_in, 2) if total_in else 0,
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\n" + "=" * 70)
    print(f"TOTAL INPUT: {total_in}   POSITIVES KEPT: {total_kept}   "
          f"({report['_summary']['kept_percent']}%)   NATIVE NEGATIVES SET ASIDE: {total_native}")
    print(f"Report: {REPORT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()