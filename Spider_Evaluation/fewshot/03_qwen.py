import os
import json
import re
import random
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SEED = 42

# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Improved Few-shot Spider SQL prediction using Qwen2.5-7B-Instruct"
)

parser.add_argument(
    "--mode",
    required=True,
    choices=["pos", "neg"],
    help="Evaluate positive or negative Spider QA",
)

parser.add_argument(
    "--train_file",
    required=True,
    help="Training JSON file used for few-shot demonstrations",
)

parser.add_argument(
    "--test_file",
    required=True,
    help="Evaluation JSON file",
)

parser.add_argument(
    "--tables_file",
    required=True,
    help="Spider tables.json file",
)

parser.add_argument(
    "--output_file",
    required=True,
    help="Output JSONL file",
)

parser.add_argument(
    "--num_shots",
    type=int,
    default=3,
    help="Number of few-shot examples. 3 is recommended for better prompt focus.",
)

parser.add_argument(
    "--shot_strategy",
    choices=["relevant", "random"],
    default="relevant",
    help="How to select few-shot examples. 'relevant' matches schema vocabulary.",
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
)

parser.add_argument(
    "--max_input_tokens",
    type=int,
    default=8192,
    help="Maximum prompt length. Keep large enough to retain the current question.",
)

parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=128,
    help="Maximum SQL generation length.",
)

parser.add_argument(
    "--seed",
    type=int,
    default=SEED,
)

parser.add_argument(
    "--do_sample",
    action="store_true",
    help="Enable stochastic generation. Default is greedy generation for accuracy.",
)

parser.add_argument(
    "--temperature",
    type=float,
    default=0.3,
)

parser.add_argument(
    "--top_p",
    type=float,
    default=0.9,
)

parser.add_argument(
    "--top_k",
    type=int,
    default=50,
)

args = parser.parse_args()

# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# ============================================================
# DEVICE
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU was not detected. "
        "Qwen2.5-7B-Instruct is not practical to run on CPU."
    )

print("=" * 70)
print("Qwen2.5-7B-Instruct Improved Few-Shot Spider Evaluation")
print("=" * 70)
print(f"CUDA device : {torch.cuda.get_device_name(0)}")
print(f"Mode        : {args.mode}")
print(f"Few-shot    : {args.num_shots}")
print(f"Shot strategy: {args.shot_strategy}")
print(f"Sampling    : {args.do_sample}")
print(f"Seed        : {args.seed}")
print()

# ============================================================
# LOAD SPIDER SCHEMAS
# ============================================================

print("Loading Spider database schemas...")

with open(args.tables_file, "r", encoding="utf-8") as f:
    tables_data = json.load(f)


def build_schema_map(tables_data):
    schema_map = {}

    for db in tables_data:
        db_id = db["db_id"]
        table_names = db["table_names_original"]
        column_names = db["column_names_original"]

        lines = []

        for table_idx, table_name in enumerate(table_names):
            columns = []

            for column in column_names:
                column_table_idx = column[0]
                column_name = column[1]

                if column_table_idx == table_idx:
                    columns.append(column_name)

            lines.append(
                f"Table {table_name}: " + ", ".join(columns)
            )

        schema_map[db_id] = "\n".join(lines)

    return schema_map


schema_map = build_schema_map(tables_data)

print(f"Loaded schemas for {len(schema_map)} databases.")
print()

# ============================================================
# LOAD DATA
# ============================================================

print("Loading few-shot training examples...")
with open(args.train_file, "r", encoding="utf-8") as f:
    train_data = json.load(f)

print(f"Training examples available: {len(train_data)}")

print("Loading evaluation data...")
with open(args.test_file, "r", encoding="utf-8") as f:
    test_data = json.load(f)

print(f"Evaluation examples: {len(test_data)}")
print()

# ============================================================
# HELPERS
# ============================================================

STOPWORDS = {
    "table", "select", "from", "where", "join", "on", "and", "or",
    "as", "the", "of", "in", "is", "are", "by", "group", "order",
    "having", "count", "distinct", "with", "not", "null"
}


def schema_tokens(schema):
    """Extract useful table/column identifiers from a schema."""
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", schema.lower())
    return {t for t in tokens if t not in STOPWORDS}


def question_tokens(question):
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", question.lower())
    return {t for t in tokens if t not in STOPWORDS and len(t) > 2}


def get_question_and_sql(example):
    return example["question"], example["sql"]


def select_shots(dataset, num_shots, target_schema, strategy, seed):
    if len(dataset) < num_shots:
        raise ValueError(
            f"Requested {num_shots} shots but only "
            f"{len(dataset)} training examples are available."
        )

    if strategy == "random":
        rng = random.Random(seed)
        return rng.sample(dataset, num_shots)

    # --------------------------------------------------------
    # Relevant-shot selection:
    # rank examples by overlap between the target schema and
    # the example's schema/question vocabulary.
    # --------------------------------------------------------

    target_tokens = schema_tokens(target_schema)

    scored = []

    for example in dataset:
        shot_db_id = example["db_id"]

        if shot_db_id not in schema_map:
            continue

        shot_schema = schema_map[shot_db_id]
        shot_question = example["question"]

        shot_schema_tokens = schema_tokens(shot_schema)
        shot_question_tokens = question_tokens(shot_question)

        schema_overlap = len(target_tokens & shot_schema_tokens)
        question_overlap = len(target_tokens & shot_question_tokens)

        # Give schema/table/column overlap the largest weight.
        score = (schema_overlap * 3) + question_overlap

        scored.append((score, example))

    if not scored:
        rng = random.Random(seed)
        return rng.sample(dataset, num_shots)

    scored.sort(key=lambda x: x[0], reverse=True)

    # Keep a small pool of highly relevant candidates and sample
    # from it so different seeds can still produce different shots.
    pool_size = min(len(scored), max(num_shots * 4, num_shots))
    candidate_pool = [x[1] for x in scored[:pool_size]]

    rng = random.Random(seed)
    return rng.sample(candidate_pool, min(num_shots, len(candidate_pool)))


# ============================================================
# PROMPT / CHAT MESSAGE CONSTRUCTION
# ============================================================

SYSTEM_PROMPT = """You are an expert Text-to-SQL system.

Convert the user's natural-language question into ONE valid SQLite SQL query using ONLY the supplied database schema.

Strict rules:
1. Return exactly ONE SQL query.
2. Return ONLY the SQL query. No explanation, reasoning, commentary, or answer text.
3. Do not output multiple alternative SQL queries.
4. Do not repeat the query.
5. Do not use markdown code fences.
6. Use only tables and columns that exist in the supplied schema.
7. Carefully preserve joins, filters, comparisons, aggregation, grouping, ordering, DISTINCT, and negation.
8. If a join is required by the schema, construct the correct join.
9. Stop immediately after the SQL query is complete.
"""

if args.mode == "neg":
    SYSTEM_PROMPT += """
10. The question may contain explicit or semantic negation. Preserve the intended negation exactly.
"""

# ============================================================
# LOAD TOKENIZER
# ============================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=os.environ.get("HF_TOKEN"),
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"

# ============================================================
# 4-BIT QUANTIZATION
# ============================================================

print("Configuring 4-bit quantization...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ============================================================
# LOAD MODEL
# ============================================================

print()
print(f"Loading model: {MODEL_ID}")
print()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
    token=os.environ.get("HF_TOKEN"),
)

model.eval()

print()
print("Model loaded successfully.")
print()

model_device = next(model.parameters()).device
print(f"Model device: {model_device}")
print()

# ============================================================
# BUILD CHAT MESSAGES
# ============================================================

def build_messages(question, db_id, shots):
    if db_id not in schema_map:
        raise KeyError(f"Schema not found for database: {db_id}")

    current_schema = schema_map[db_id]

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    # Each demonstration is a proper user -> assistant pair.
    # This is much better for an Instruct model than putting
    # "Question/SQL" examples into one long plain-text prompt.
    for i, example in enumerate(shots, start=1):
        shot_db_id = example["db_id"]

        if shot_db_id not in schema_map:
            continue

        shot_question, shot_sql = get_question_and_sql(example)
        shot_schema = schema_map[shot_db_id]

        demo_user = (
            f"Example {i}\n\n"
            f"Database: {shot_db_id}\n\n"
            f"Schema:\n{shot_schema}\n\n"
            f"Question:\n{shot_question}\n\n"
            "Return only the SQL query."
        )

        messages.append({
            "role": "user",
            "content": demo_user,
        })

        messages.append({
            "role": "assistant",
            "content": shot_sql,
        })

    current_user = (
        f"Database: {db_id}\n\n"
        f"Schema:\n{current_schema}\n\n"
        f"Question:\n{question}\n\n"
        "Return exactly one SQL query and nothing else."
    )

    messages.append({
        "role": "user",
        "content": current_user,
    })

    return messages


# ============================================================
# SQL CLEANING
# ============================================================

def split_sql_statements(text):
    """Split SQL on semicolons outside quoted strings."""
    statements = []
    start = 0
    quote = None
    escaped = False

    for i, ch in enumerate(text):
        if escaped:
            escaped = False
            continue

        if ch == "\\" and quote:
            escaped = True
            continue

        if quote:
            if ch == quote:
                quote = None
            continue

        if ch in ("'", '"', "`"):
            quote = ch
            continue

        if ch == ";":
            statements.append(text[start:i + 1].strip())
            start = i + 1

    tail = text[start:].strip()
    if tail:
        statements.append(tail)

    return [s for s in statements if s]


def clean_sql(text):
    text = text.strip()

    # Remove markdown fences.
    text = re.sub(r"```(?:sql)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "").strip()

    # Remove common explanation prefixes.
    text = re.sub(
        r"^\s*(SQL\s*Query|SQL|Answer)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    # If explanation appears after the SQL, discard it.
    text = re.split(
        r"\n\s*(Explanation|Reasoning|Reason|Answer|Here is|Here’s)\s*[:\-]?",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()

    # Find the beginning of the SQL.
    match = re.search(r"\b(SELECT|WITH)\b", text, flags=re.IGNORECASE)
    if match:
        text = text[match.start():].strip()

    # Keep only the first SQL statement if the model generated
    # several alternatives.
    statements = split_sql_statements(text)

    if statements:
        text = statements[0]

    # Remove trailing semicolon for cleaner exact-match comparison.
    text = text.rstrip().rstrip(";").strip()

    return text


def normalize_sql(sql):
    sql = clean_sql(sql)
    sql = re.sub(r"\s+", " ", sql)
    return sql.lower().strip()


# ============================================================
# GENERATION
# ============================================================

@torch.inference_mode()
def generate_sql(messages):
    # Use the native Llama-3.1 Instruct chat template.
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_input_tokens,
    )

    inputs = {
        key: value.to(model_device)
        for key, value in inputs.items()
    }

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "num_beams": 1,
        "repetition_penalty": 1.0,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
    }

    if args.do_sample:
        generation_kwargs.update({
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        })
    else:
        # Greedy decoding is the recommended default for
        # Spider SQL exact-match evaluation.
        generation_kwargs["do_sample"] = False

    outputs = model.generate(
        **inputs,
        **generation_kwargs,
    )

    input_length = inputs["input_ids"].shape[1]

    generated_tokens = outputs[0, input_length:]

    prediction = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return clean_sql(prediction)


# ============================================================
# OUTPUT FILE
# ============================================================

output_path = Path(args.output_file)
output_path.parent.mkdir(parents=True, exist_ok=True)

# ============================================================
# RESUME SUPPORT
# ============================================================

processed_indices = set()

if output_path.exists():
    print(f"Existing output found: {output_path}")
    print("Checking already processed examples...")

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

                if "index" in record:
                    processed_indices.add(record["index"])

            except json.JSONDecodeError:
                print("Warning: invalid JSONL line found. Skipping it.")

    print(f"Already processed: {len(processed_indices)} examples")

# ============================================================
# RUN EVALUATION
# ============================================================

print("=" * 70)
print("Starting inference")
print("=" * 70)

correct_em = 0
total_processed = 0

with open(output_path, "a", encoding="utf-8") as output_file:

    for idx, example in enumerate(
        tqdm(test_data, desc="Generating SQL")
    ):

        if idx in processed_indices:
            continue

        db_id = example["db_id"]
        question = example["question"]
        gold_sql = example["sql"]

        # Select relevant demonstrations for THIS question.
        # This is intentionally done per test question rather
        # than once globally.
        target_schema = schema_map[db_id]

        shots = select_shots(
            train_data,
            args.num_shots,
            target_schema,
            args.shot_strategy,
            args.seed + idx,
        )

        messages = build_messages(
            question=question,
            db_id=db_id,
            shots=shots,
        )

        try:
            predicted_sql = generate_sql(messages)
            error = None

        except Exception as e:
            predicted_sql = ""
            error = str(e)

        if predicted_sql:
            is_exact_match = (
                normalize_sql(predicted_sql)
                == normalize_sql(gold_sql)
            )
        else:
            is_exact_match = False

        result = {
            "index": idx,
            "db_id": db_id,
            "mode": args.mode,
            "question": question,
            "gold_sql": gold_sql,
            "predicted_sql": predicted_sql,
            "exact_match": is_exact_match,
            "error": error,
            "seed": args.seed,
            "num_shots": args.num_shots,
            "shot_strategy": args.shot_strategy,
            "do_sample": args.do_sample,
        }

        output_file.write(
            json.dumps(
                result,
                ensure_ascii=False,
            ) + "\n"
        )

        output_file.flush()
        os.fsync(output_file.fileno())

        total_processed += 1

        if is_exact_match:
            correct_em += 1

# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print(f"Model           : {MODEL_ID}")
print(f"Mode            : {args.mode}")
print(f"Few-shot        : {args.num_shots}")
print(f"Shot strategy   : {args.shot_strategy}")
print(f"Sampling        : {args.do_sample}")
print(f"Seed            : {args.seed}")
print(f"New predictions : {total_processed}")
print(f"Correct EM      : {correct_em}")
print(f"Output          : {output_path}")
print("=" * 70)
