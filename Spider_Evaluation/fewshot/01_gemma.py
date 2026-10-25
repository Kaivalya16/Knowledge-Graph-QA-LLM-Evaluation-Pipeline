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

MODEL_ID = "google/gemma-2-9b-it"

SEED = 42

# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Few-shot Spider SQL prediction using Gemma-2-9B-IT"
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
    help="Training JSON file used only for few-shot demonstrations",
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
    help="Output JSON file",
)

parser.add_argument(
    "--num_shots",
    type=int,
    default=5,
    help="Number of few-shot examples",
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="Batch size. For Gemma-2-9B use 1 initially.",
)

parser.add_argument(
    "--max_input_tokens",
    type=int,
    default=4096,
)

parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=256,
)

parser.add_argument(
    "--seed",
    type=int,
    default=SEED,
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
        "Gemma-2-9B-IT is not practical to run on CPU."
    )

print("=" * 70)
print("Gemma-2-9B-IT Few-Shot Spider Evaluation")
print("=" * 70)

print(f"CUDA device : {torch.cuda.get_device_name(0)}")
print(f"Mode        : {args.mode}")
print(f"Few-shot    : {args.num_shots}")
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
                f"Table {table_name}: "
                + ", ".join(columns)
            )

        schema_map[db_id] = "\n".join(lines)

    return schema_map


schema_map = build_schema_map(tables_data)

print(f"Loaded schemas for {len(schema_map)} databases.")
print()


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("Loading few-shot training examples...")

with open(args.train_file, "r", encoding="utf-8") as f:
    train_data = json.load(f)

print(f"Training examples available: {len(train_data)}")


# ============================================================
# LOAD TEST DATA
# ============================================================

print("Loading evaluation data...")

with open(args.test_file, "r", encoding="utf-8") as f:
    test_data = json.load(f)

print(f"Evaluation examples: {len(test_data)}")
print()


# ============================================================
# SELECT FEW-SHOT EXAMPLES
# ============================================================

def select_shots(dataset, num_shots):

    if len(dataset) < num_shots:
        raise ValueError(
            f"Requested {num_shots} shots but only "
            f"{len(dataset)} training examples are available."
        )

    return random.sample(dataset, num_shots)


shots = select_shots(
    train_data,
    args.num_shots
)

print("Selected few-shot demonstrations:")

for i, example in enumerate(shots, start=1):

    if args.mode == "pos":
        question = example["question"]
        sql = example["sql"]

    else:
        question = example["question"]
        sql = example["sql"]

    print(f"\n--- Shot {i} ---")
    print("DB       :", example["db_id"])
    print("Question :", question)
    print("SQL      :", sql)

print()


# ============================================================
# PROMPT CONSTRUCTION
# ============================================================

def get_question_and_sql(example, mode):

    if mode == "pos":

        question = example["question"]
        sql = example["sql"]

    else:

        question = example["question"]
        sql = example["sql"]

    return question, sql


def build_prompt(
    question,
    db_id,
    shots,
    mode,
):

    if db_id not in schema_map:
        raise KeyError(
            f"Schema not found for database: {db_id}"
        )

    current_schema = schema_map[db_id]

    prompt = ""

    # --------------------------------------------------------
    # SYSTEM INSTRUCTION
    # --------------------------------------------------------

    prompt += (
        "You are an expert Text-to-SQL system.\n"
        "Your task is to convert natural language questions "
        "into valid SQLite SQL queries.\n\n"
    )

    prompt += (
        "Follow these rules:\n"
        "1. Use only tables and columns from the provided schema.\n"
        "2. Carefully understand the meaning of the question.\n"
        "3. Preserve negation, comparison, aggregation, joins, "
        "ordering, grouping, and filtering exactly as requested.\n"
        "4. Return ONLY the SQL query.\n"
        "5. Do not provide explanations.\n"
        "6. Do not use markdown code fences.\n\n"
    )

    if mode == "neg":

        prompt += (
            "The evaluation question may contain explicit or "
            "semantic negation. Pay particular attention to "
            "words such as not, isn't, aren't, without, "
            "other than, less than, greater than, etc.\n\n"
        )

    # --------------------------------------------------------
    # FEW-SHOT EXAMPLES
    # --------------------------------------------------------

    prompt += "Here are examples of how to solve the task.\n\n"

    for i, example in enumerate(shots, start=1):

        shot_db_id = example["db_id"]

        shot_question, shot_sql = get_question_and_sql(
            example,
            mode
        )

        if shot_db_id not in schema_map:
            continue

        shot_schema = schema_map[shot_db_id]

        prompt += f"Example {i}\n"

        prompt += (
            f"Database: {shot_db_id}\n\n"
            f"Schema:\n"
            f"{shot_schema}\n\n"
            f"Question:\n"
            f"{shot_question}\n\n"
            f"SQL:\n"
            f"{shot_sql}\n\n"
        )

    # --------------------------------------------------------
    # CURRENT QUESTION
    # --------------------------------------------------------

    prompt += "Now solve the following question.\n\n"

    prompt += (
        f"Database: {db_id}\n\n"
        f"Schema:\n"
        f"{current_schema}\n\n"
        f"Question:\n"
        f"{question}\n\n"
        f"SQL:\n"
    )

    return prompt


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
print("If this is the first run, Hugging Face will download it.")
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


# ============================================================
# MODEL INPUT DEVICE
# ============================================================

model_device = next(model.parameters()).device

print(f"Model device: {model_device}")
print()


# ============================================================
# CLEAN SQL
# ============================================================

def clean_sql(text):

    text = text.strip()

    # Remove markdown code fences
    text = re.sub(
        r"```(?:sql)?",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = text.replace("```", "")

    text = text.strip()

    # If model generated explanations after SQL,
    # keep only the SQL-looking part.
    lines = text.splitlines()

    sql_lines = []

    started = False

    for line in lines:

        stripped = line.strip()

        if not stripped:
            continue

        upper = stripped.upper()

        if (
            upper.startswith("SELECT ")
            or upper.startswith("WITH ")
        ):
            started = True

        if started:
            sql_lines.append(stripped)

    if sql_lines:
        text = " ".join(sql_lines)

    # Remove accidental final explanation
    # in common cases.
    text = re.split(
        r"\n\s*(Explanation|Reason|Answer)\s*:",
        text,
        flags=re.IGNORECASE
    )[0]

    return text.strip()


# ============================================================
# NORMALIZE SQL
# ============================================================

def normalize_sql(sql):

    sql = sql.strip()

    sql = re.sub(
        r"\s+",
        " ",
        sql
    )

    return sql.lower()


# ============================================================
# GENERATE SINGLE PREDICTION
# ============================================================

@torch.inference_mode()
def generate_sql(prompt):

    inputs = tokenizer(

        prompt,

        return_tensors="pt",

        truncation=True,

        max_length=args.max_input_tokens,

    )

    inputs = {
        key: value.to(model_device)
        for key, value in inputs.items()
    }

    outputs = model.generate(

        **inputs,

        max_new_tokens=args.max_new_tokens,

        do_sample=True,

        temperature=0.7,

        top_p=0.9,

        top_k=50,

        num_beams=1,

        repetition_penalty=1.05,

        eos_token_id=tokenizer.eos_token_id,

        pad_token_id=tokenizer.pad_token_id,

        use_cache=True,

    )

    # Decode only newly generated tokens
    input_length = inputs["input_ids"].shape[1]

    generated_tokens = outputs[
        0,
        input_length:
    ]

    prediction = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return clean_sql(prediction)


# ============================================================
# OUTPUT FILE
# ============================================================

output_path = Path(args.output_file)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# RESUME SUPPORT
# ============================================================

processed_indices = set()

if output_path.exists():

    print(f"Existing output found: {output_path}")
    print("Checking already processed examples...")

    with open(
        output_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

                if "index" in record:
                    processed_indices.add(
                        record["index"]
                    )

            except json.JSONDecodeError:
                print(
                    "Warning: invalid JSONL line found. "
                    "Skipping it."
                )

    print(
        f"Already processed: "
        f"{len(processed_indices)} examples"
    )


# ============================================================
# RUN EVALUATION
# ============================================================

print("=" * 70)
print("Starting inference")
print("=" * 70)

correct_em = 0
total_processed = 0


with open(
    output_path,
    "a",
    encoding="utf-8"
) as output_file:

    for idx, example in enumerate(
        tqdm(test_data, desc="Generating SQL")
    ):

        # ----------------------------------------------------
        # SKIP ALREADY PROCESSED EXAMPLES
        # ----------------------------------------------------

        if idx in processed_indices:
            continue

        db_id = example["db_id"]

        # ----------------------------------------------------
        # SELECT QUESTION + GOLD SQL
        # ----------------------------------------------------

        if args.mode == "pos":

            question = example["question"]
            gold_sql = example["sql"]

        else:

            question = example["question"]
            gold_sql = example["sql"]

        # ----------------------------------------------------
        # BUILD PROMPT
        # ----------------------------------------------------

        prompt = build_prompt(
            question=question,
            db_id=db_id,
            shots=shots,
            mode=args.mode,
        )

        # ----------------------------------------------------
        # GENERATE
        # ----------------------------------------------------

        try:

            predicted_sql = generate_sql(prompt)

            error = None

        except Exception as e:

            predicted_sql = ""

            error = str(e)

        # ----------------------------------------------------
        # EXACT MATCH
        # ----------------------------------------------------

        if predicted_sql:

            is_exact_match = (
                normalize_sql(predicted_sql)
                == normalize_sql(gold_sql)
            )

        else:

            is_exact_match = False

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        result = {

            "index": idx,

            "db_id": db_id,

            "mode": args.mode,

            "question": question,

            "gold_sql": gold_sql,

            "predicted_sql": predicted_sql,

            "exact_match": is_exact_match,

            "error": error,

        }

        # ----------------------------------------------------
        # WRITE IMMEDIATELY
        # ----------------------------------------------------

        output_file.write(
            json.dumps(
                result,
                ensure_ascii=False
            ) + "\n"
        )

        # ----------------------------------------------------
        # FORCE WRITE TO DISK
        # ----------------------------------------------------

        output_file.flush()

        # This makes sure the result is physically written
        # instead of remaining only in Python's buffer.
        os.fsync(
            output_file.fileno()
        )

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

print(f"Model          : {MODEL_ID}")
print(f"Mode           : {args.mode}")
print(f"Few-shot       : {args.num_shots}")
print(f"New predictions: {total_processed}")
print(f"Output         : {output_path}")

print("=" * 70)