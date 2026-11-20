"""
Shared prompt and data helpers for the Spider positive/negative Text-to-SQL
experiments.

Fine-tuning, inference and few-shot scripts all import this module, so the
model sees exactly the same prompt at training and at test time. v1 copied
the prompt code into each script, which lets the copies drift apart.

Prompt variants
  base      the original rules
  negation  base + rules 8-13 on negation scope (default)
Select with build_messages(..., prompt_variant=...) or T2S_PROMPT_VARIANT.
"""

import hashlib
import json
import os
import re


BASE_PROMPT = """You are an expert Text-to-SQL system.

Convert the natural language question into a valid SQLite SQL query.

Rules:
1. Use only the tables and columns present in the provided schema.
2. Do not invent tables or columns.
3. Use a JOIN only when the question requires information from multiple tables
   or when the schema requires a relationship between tables.
4. Preserve all filtering, comparison, aggregation, ordering, grouping,
   join, and negation requirements.
5. Return only the SQL query.
6. Do not provide explanations.
7. Do not use markdown code fences."""


# The negation rules follow the conventions the gold SQL was repaired with
# (fix_negative_sql.py), so the prompt never asks for SQL the labels contradict.
# NULL handling is deliberately not mentioned: the gold set mostly keeps plain
# != semantics, and a NULL rule would push the model away from the labels.
NEGATION_RULES = """

Negation (check silently; still output only the SQL query):
8. Find every negation in the question: "not", "no", "never", "none",
   "without", "except", "other than", "excluding", "n't", "no more than",
   "not at least", "not before", and similar. "not only ... but also" is not a
   negation. Every negation must appear in the SQL, and the final query must
   have the polarity the question asks for.
9. A negation on a value of the row being returned negates that condition:
   "not X" -> col != X, NOT LIKE, NOT IN, NOT BETWEEN;
   "not more than N" -> col <= N; "not less than N" -> col >= N;
   "not before Y" -> col >= Y; "not after Y" -> col <= Y.
10. A negation saying an entity has no related row of some kind
   ("students not enrolled in X", "movies not reviewed by Y",
   "customers who do not have orders") must exclude the entities that do have
   such a row. Do not put != on the joined table: that keeps entities with any
   other related row. Use
   ... WHERE entity.key NOT IN (SELECT key FROM ... WHERE <condition>)
   or <all entities> EXCEPT <entities with the condition>.
11. A negation on an indefinite noun ("a / some / any X that is not Y",
   "an X not named Y") only requires that related row not to match:
   use != on that row.
12. "both X and not Y" / "X but not Y" -> <entities with X> EXCEPT
   <entities with Y>. Never INTERSECT with a negated condition.
13. "no more than N", "not at least N" or "do not have N or more" related rows
   includes entities with zero related rows, which JOIN ... GROUP BY ...
   HAVING count(*) < N can never return. Use
   ... WHERE entity.key NOT IN (SELECT key ... GROUP BY key HAVING count(*) >= N)."""


PROMPT_VARIANTS = {
    "base": BASE_PROMPT,
    "negation": BASE_PROMPT + NEGATION_RULES,
}

# Choose the variant once, here or with the T2S_PROMPT_VARIANT environment
# variable. Training and inference MUST use the same variant; log
# prompt_id() with every run and adapter to make that checkable.
DEFAULT_PROMPT_VARIANT = os.environ.get("T2S_PROMPT_VARIANT", "negation")
if DEFAULT_PROMPT_VARIANT not in PROMPT_VARIANTS:
    raise ValueError(f"T2S_PROMPT_VARIANT must be one of {sorted(PROMPT_VARIANTS)}, "
                     f"got '{DEFAULT_PROMPT_VARIANT}'")

# kept for scripts that import SYSTEM_PROMPT directly
SYSTEM_PROMPT = PROMPT_VARIANTS[DEFAULT_PROMPT_VARIANT]


def get_system_prompt(variant=None):
    variant = variant or DEFAULT_PROMPT_VARIANT
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"Unknown prompt variant '{variant}'. Options: {sorted(PROMPT_VARIANTS)}")
    return PROMPT_VARIANTS[variant]


def prompt_id(variant=None):
    """e.g. 'negation-3f9a1c2b': variant name plus a hash of the exact text.
    Save it next to each adapter and in each prediction file."""
    variant = variant or DEFAULT_PROMPT_VARIANT
    digest = hashlib.sha256(get_system_prompt(variant).encode("utf-8")).hexdigest()[:8]
    return f"{variant}-{digest}"


# ------------------------------------------------------------
# SCHEMA
# ------------------------------------------------------------

def load_schema_map(tables_file, with_types=True, with_keys=True):
    """db_id -> schema text built from Spider's tables.json.

    Primary and foreign keys are included because they tell the model how
    tables join; v1 listed column names only.
    """
    with open(tables_file, "r", encoding="utf-8") as handle:
        tables = json.load(handle)

    schema_map = {}
    for db in tables:
        table_names = db["table_names_original"]
        columns = db["column_names_original"]
        types = db.get("column_types", ["text"] * len(columns))
        primary = set()
        for key in db.get("primary_keys", []):
            primary.update(key if isinstance(key, list) else [key])

        per_table = {name: [] for name in table_names}
        for index, (table_index, column_name) in enumerate(columns):
            if table_index == -1:
                continue
            details = []
            if with_types:
                details.append(types[index])
            if with_keys and index in primary:
                details.append("primary key")
            text = column_name + (f" ({', '.join(details)})" if details else "")
            per_table[table_names[table_index]].append(text)

        lines = [f"Table {name}: {', '.join(per_table[name])}" for name in table_names]

        if with_keys and db.get("foreign_keys"):
            lines.append("Foreign keys:")
            for source, target in db["foreign_keys"]:
                s_table, s_column = columns[source]
                t_table, t_column = columns[target]
                lines.append(
                    f"  {table_names[s_table]}.{s_column} = "
                    f"{table_names[t_table]}.{t_column}"
                )

        schema_map[db["db_id"]] = "\n".join(lines)
    return schema_map


# ------------------------------------------------------------
# FIELDS
# ------------------------------------------------------------

QUESTION_KEYS = ("question", "negative_question", "negated_question")
SQL_KEYS = ("sql", "negative_sql", "negated_sql", "query")


def get_question(example):
    """The final pos_*/neg_* files both store the question in "question";
    older formats are still accepted."""
    for key in QUESTION_KEYS:
        if example.get(key):
            return example[key]
    raise KeyError(f"No question field. Keys: {list(example)}")


def get_sql(example):
    for key in SQL_KEYS:
        if example.get(key):
            return example[key]
    raise KeyError(f"No SQL field. Keys: {list(example)}")


def normalize_sql(sql):
    """Strip fences and a trailing ';', collapse whitespace OUTSIDE string
    literals (Spider gold SQL often has "a  =  b")."""
    sql = str(sql).strip()
    sql = re.sub(r"```(?:sql)?", "", sql, flags=re.IGNORECASE).replace("```", "").strip()
    parts = re.split(r"('(?:[^']|'')*'|\"[^\"]*\")", sql)
    for index in range(0, len(parts), 2):
        parts[index] = re.sub(r"\s+", " ", parts[index])
    return "".join(parts).strip().rstrip(";").strip()


# ------------------------------------------------------------
# MESSAGES
# ------------------------------------------------------------

def build_user_content(example, schema_map):
    db_id = example["db_id"]
    if db_id not in schema_map:
        raise KeyError(f"Database '{db_id}' is not present in the tables file.")
    return (
        f"Database:\n{db_id}\n\n"
        f"Schema:\n{schema_map[db_id]}\n\n"
        f"Question:\n{get_question(example)}"
    )


def build_messages(example, schema_map, with_answer=False, few_shot=None, prompt_variant=None):
    """Chat messages. `few_shot` is an optional list of solved examples,
    inserted as earlier user/assistant turns (used by the few-shot scripts).
    `prompt_variant` ("base" or "negation") defaults to DEFAULT_PROMPT_VARIANT."""
    messages = [{"role": "system", "content": get_system_prompt(prompt_variant)}]
    for shot in few_shot or []:
        messages.append({"role": "user", "content": build_user_content(shot, schema_map)})
        messages.append({"role": "assistant", "content": normalize_sql(get_sql(shot))})
    messages.append({"role": "user", "content": build_user_content(example, schema_map)})
    if with_answer:
        messages.append({"role": "assistant", "content": normalize_sql(get_sql(example))})
    return messages


# ------------------------------------------------------------
# OUTPUT CLEANING
# ------------------------------------------------------------

def extract_sql(text):
    """First SQL statement from a model response."""
    text = (text or "").strip()
    text = re.sub(r"```(?:sql)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()
    for prefix in ("SQL:", "sql:", "SQLite:", "Answer:", "answer:", "Query:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    start = re.search(r"\b(SELECT|WITH)\b", text, re.IGNORECASE)
    if start and start.start() > 0:
        text = text[start.start():]
    # first statement only; a ';' inside a string literal is kept
    parts = re.split(r"('(?:[^']|'')*'|\"[^\"]*\")", text)
    out = []
    for index, part in enumerate(parts):
        if index % 2 == 0 and ";" in part:
            out.append(part.split(";", 1)[0])
            break
        out.append(part)
    text = "".join(out)
    text = text.split("\n\n", 1)[0]   # drop trailing explanations
    return normalize_sql(text)