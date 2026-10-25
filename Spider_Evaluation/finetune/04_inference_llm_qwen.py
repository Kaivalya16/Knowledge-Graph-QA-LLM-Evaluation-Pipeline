import os
import json
import argparse
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import PeftModel

# ============================================================
# MODEL
# ============================================================

MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# LOAD TABLES
# ============================================================

def load_tables(tables_file):

    tables = load_json(
        tables_file
    )

    lookup = {}

    for db in tables:

        lookup[
            db["db_id"]
        ] = db

    return lookup


# ============================================================
# BUILD SCHEMA
# ============================================================

def build_schema(
    db_id,
    tables_lookup
):

    db = tables_lookup[
        db_id
    ]

    table_names = (
        db["table_names_original"]
    )

    column_names = (
        db["column_names_original"]
    )

    table_columns = {
        i: []
        for i in range(
            len(table_names)
        )
    }

    for table_id, column_name in column_names:

        if table_id == -1:
            continue

        table_columns[
            table_id
        ].append(
            column_name
        )

    schema_parts = []

    for table_id, table_name in enumerate(
        table_names
    ):

        columns = table_columns[
            table_id
        ]

        schema_parts.append(
            f"Table {table_name}:\n"
            + "\n".join(
                f"  - {column}"
                for column in columns
            )
        )

    return "\n\n".join(
        schema_parts
    )


# ============================================================
# GET QUESTION
# ============================================================

def get_question(
    example,
    mode
):

    if mode == "pos":

        return example[
            "question"
        ]

    elif mode == "neg":

        return example[
            "question"
        ]

    raise ValueError(
        "mode must be 'pos' or 'neg'"
    )


# ============================================================
# BUILD PROMPT
#
# MUST MATCH THE FINE-TUNING SCRIPT
# ============================================================

def build_prompt(
    db_id,
    schema,
    question
):

    return f"""You are an expert Text-to-SQL system.

Convert the natural language question into a valid SQLite SQL query.

Rules:
1. Use only the tables and columns present in the schema.
2. Carefully understand the meaning of the question.
3. Preserve all filtering, comparison, aggregation, ordering,
   grouping, joins, and negation requirements.
4. Return only the SQL query.
5. Do not provide explanations.
6. Do not use markdown code fences.

Database:
{db_id}

Schema:
{schema}

Question:
{question}

SQL:"""


# ============================================================
# CLEAN GENERATED SQL
# ============================================================

def clean_sql(text):

    text = text.strip()

    # --------------------------------------------------------
    # Remove markdown fences
    # --------------------------------------------------------

    text = text.replace(
        "```sql",
        ""
    )

    text = text.replace(
        "```SQL",
        ""
    )

    text = text.replace(
        "```",
        ""
    )

    text = text.strip()

    # --------------------------------------------------------
    # Remove common prefixes
    # --------------------------------------------------------

    prefixes = [
        "SQL:",
        "sql:",
        "SQL Query:",
        "sql query:",
    ]

    for prefix in prefixes:

        if text.startswith(
            prefix
        ):

            text = text[
                len(prefix):
            ].strip()

    # --------------------------------------------------------
    # If model generated explanation after SQL,
    # keep the SQL statement.
    # --------------------------------------------------------

    if ";" in text:

        text = (
            text.split(
                ";",
                1
            )[0]
            .strip()
            + ";"
        )

    return text


# ============================================================
# LOAD MODEL + LORA
# ============================================================

def load_model(
    adapter_dir
):

    print()
    print("=" * 60)
    print("Loading Qwen2.5-32B-Instruct")
    print("=" * 60)

    print(
        f"Base model: {MODEL_ID}"
    )

    print(
        f"LoRA adapter: {adapter_dir}"
    )

    print(
        "Quantization: 4-bit NF4"
    )

    print()

    # --------------------------------------------------------
    # 4-bit configuration
    # --------------------------------------------------------

    bnb_config = BitsAndBytesConfig(

        load_in_4bit=True,

        bnb_4bit_quant_type="nf4",

        bnb_4bit_compute_dtype=torch.bfloat16,

        bnb_4bit_use_double_quant=True,
    )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    # --------------------------------------------------------
    # Base model
    # --------------------------------------------------------

    model = AutoModelForCausalLM.from_pretrained(

        MODEL_ID,

        quantization_config=bnb_config,

        torch_dtype=torch.bfloat16,

        device_map={
            "": 0
        },

        trust_remote_code=True,
    )

    # --------------------------------------------------------
    # Load LoRA adapter
    # --------------------------------------------------------

    print()
    print("Loading LoRA adapter...")

    model = PeftModel.from_pretrained(

        model,

        adapter_dir,

        is_trainable=False,
    )

    model.eval()

    print()
    print("Model loaded successfully.")
    print()

    return model, tokenizer


# ============================================================
# GENERATE SQL
# ============================================================

@torch.inference_mode()
def generate_sql(
    model,
    tokenizer,
    prompt,
    max_input_tokens,
    max_new_tokens
):

    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    inputs = tokenizer(

        prompt,

        return_tensors="pt",

        truncation=True,

        max_length=max_input_tokens,
    )

    # --------------------------------------------------------
    # Move inputs to model device
    # --------------------------------------------------------

    inputs = {
        key: value.to(
            model.device
        )
        for key, value in inputs.items()
    }

    input_length = (
        inputs["input_ids"]
        .shape[1]
    )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    outputs = model.generate(

        **inputs,

        max_new_tokens=max_new_tokens,

        do_sample=False,

        num_beams=1,

        repetition_penalty=1.0,

        eos_token_id=(
            tokenizer.eos_token_id
        ),

        pad_token_id=(
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        ),

        use_cache=True,
    )

    # --------------------------------------------------------
    # Only decode newly generated tokens
    # --------------------------------------------------------

    generated_tokens = outputs[
        0,
        input_length:
    ]

    generated_text = tokenizer.decode(

        generated_tokens,

        skip_special_tokens=True,
    )

    return clean_sql(
        generated_text
    )


# ============================================================
# LOAD EXISTING INDICES
# ============================================================

def load_existing_indices(
    output_file
):

    existing_indices = set()

    if not os.path.exists(
        output_file
    ):

        return existing_indices

    print(
        f"Existing output found: "
        f"{output_file}"
    )

    with open(
        output_file,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:

                record = json.loads(
                    line
                )

                if "index" in record:

                    existing_indices.add(
                        str(
                            record["index"]
                        )
                    )

            except json.JSONDecodeError:

                continue

    print(
        f"Already completed: "
        f"{len(existing_indices)}"
    )

    return existing_indices


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Qwen2.5-32B-Instruct "
            "QLoRA inference for Spider"
        )
    )

    # --------------------------------------------------------
    # Mode
    # --------------------------------------------------------

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "pos",
            "neg"
        ],
        help="Positive or negative inference",
    )

    # --------------------------------------------------------
    # Test file
    # --------------------------------------------------------

    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Test JSON file",
    )

    # --------------------------------------------------------
    # Tables
    # --------------------------------------------------------

    parser.add_argument(
        "--tables_file",
        type=str,
        required=True,
        help="Spider tables.json",
    )

    # --------------------------------------------------------
    # LoRA adapter
    # --------------------------------------------------------

    parser.add_argument(
        "--adapter_dir",
        type=str,
        required=True,
        help="Path to trained LoRA adapter",
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output JSONL file",
    )

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=4096,
        help="Maximum input tokens",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum generated tokens",
    )

    # --------------------------------------------------------
    # Optional start index
    # --------------------------------------------------------

    parser.add_argument(
        "--start_index",
        type=int,
        default=None,
        help="Start from this dataset index",
    )

    # --------------------------------------------------------
    # Optional number of questions
    # --------------------------------------------------------

    parser.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="Maximum questions to process",
    )

    args = parser.parse_args()

    # ========================================================
    # LOAD TABLES
    # ========================================================

    print()
    print("Loading Spider tables...")

    tables_lookup = load_tables(
        args.tables_file
    )

    print(
        f"Databases loaded: "
        f"{len(tables_lookup)}"
    )

    # ========================================================
    # LOAD TEST DATA
    # ========================================================

    print()
    print("Loading test data...")

    test_data = load_json(
        args.test_file
    )

    print(
        f"Test examples: "
        f"{len(test_data)}"
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    model, tokenizer = load_model(
        args.adapter_dir
    )

    # ========================================================
    # FIND ALREADY PROCESSED
    # ========================================================

    existing_indices = (
        load_existing_indices(
            args.output_file
        )
    )

    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    output_dir = os.path.dirname(
        args.output_file
    )

    if output_dir:

        os.makedirs(
            output_dir,
            exist_ok=True
        )

    # ========================================================
    # SELECT QUESTIONS
    # ========================================================

    questions_to_process = []

    for position, example in enumerate(
        test_data
    ):

        # ----------------------------------------------------
        # Dataset index
        # ----------------------------------------------------

        if "index" in example:

            index = example[
                "index"
            ]

        else:

            index = position

        # ----------------------------------------------------
        # Start index
        # ----------------------------------------------------

        if (
            args.start_index is not None
            and index < args.start_index
        ):

            continue

        # ----------------------------------------------------
        # Resume
        # ----------------------------------------------------

        if str(index) in existing_indices:

            continue

        questions_to_process.append(
            (
                position,
                index,
                example
            )
        )

        # ----------------------------------------------------
        # Limit questions
        # ----------------------------------------------------

        if (
            args.max_questions is not None
            and
            len(questions_to_process)
            >= args.max_questions
        ):

            break

    print()
    print(
        f"Questions to process: "
        f"{len(questions_to_process)}"
    )

    # ========================================================
    # OPEN OUTPUT
    # ========================================================

    output_file = open(
        args.output_file,
        "a",
        encoding="utf-8"
    )

    # ========================================================
    # INFERENCE
    # ========================================================

    successful = 0
    failed = 0

    total = len(
        questions_to_process
    )

    try:

        for count, (
            position,
            index,
            example
        ) in enumerate(
            questions_to_process,
            1
        ):

            try:

                db_id = example[
                    "db_id"
                ]

                question = get_question(
                    example,
                    args.mode
                )

                schema = build_schema(
                    db_id,
                    tables_lookup
                )

                # ------------------------------------------------
                # Build same prompt used during fine-tuning
                # ------------------------------------------------

                prompt = build_prompt(

                    db_id,

                    schema,

                    question
                )

                # ------------------------------------------------
                # Generate SQL
                # ------------------------------------------------

                predicted_sql = generate_sql(

                    model,

                    tokenizer,

                    prompt,

                    args.max_input_tokens,

                    args.max_new_tokens
                )

                # ------------------------------------------------
                # Create result
                # ------------------------------------------------

                result = {

                    "index": index,

                    "db_id": db_id,

                    "question": question,

                    "predicted_sql":
                        predicted_sql,
                }

                # ------------------------------------------------
                # Write immediately
                # ------------------------------------------------

                output_file.write(
                    json.dumps(
                        result,
                        ensure_ascii=False
                    )
                    + "\n"
                )

                output_file.flush()

                os.fsync(
                    output_file.fileno()
                )

                successful += 1

                percent = (
                    100 * count / total
                    if total > 0
                    else 100
                )

                print(
                    f"\rProgress: "
                    f"{count}/{total} "
                    f"({percent:.2f}%) "
                    f"| index={index}",
                    end="",
                    flush=True
                )

            except Exception as e:

                failed += 1

                print()

                print(
                    f"[ERROR] "
                    f"index={index}: "
                    f"{type(e).__name__}: "
                    f"{e}"
                )

                # ------------------------------------------------
                # Save failed example too
                # ------------------------------------------------

                result = {

                    "index": index,

                    "db_id":
                        example.get(
                            "db_id"
                        ),

                    "question":
                        get_question(
                            example,
                            args.mode
                        ),

                    "predicted_sql":
                        None,

                    "error":
                        str(e),
                }

                output_file.write(
                    json.dumps(
                        result,
                        ensure_ascii=False
                    )
                    + "\n"
                )

                output_file.flush()

                os.fsync(
                    output_file.fileno()
                )

    finally:

        output_file.close()

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print()
    print("=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)

    print(
        f"Processed:       {total}"
    )

    print(
        f"Successful:      {successful}"
    )

    print(
        f"Failed:          {failed}"
    )

    print(
        f"Output file:     {args.output_file}"
    )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()