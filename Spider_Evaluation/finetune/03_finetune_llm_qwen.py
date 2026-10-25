import os
# Reduce CUDA allocator fragmentation on small-VRAM systems.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

import json
import argparse
import torch
import torch.nn.functional as F

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

from peft import (
    LoraConfig,
    get_peft_model,
)


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
# LOAD SPIDER TABLES
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
# BUILD DATABASE SCHEMA
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
# GET SQL
# ============================================================

def get_sql(
    example,
    mode
):

    if mode == "pos":

        if "sql" in example:

            return example[
                "sql"
            ]

        if "query" in example:

            return example[
                "query"
            ]

        raise KeyError(
            "Positive example does not contain "
            "'sql' or 'query'."
        )

    elif mode == "neg":

        return example[
            "sql"
        ]

    raise ValueError(
        "mode must be 'pos' or 'neg'"
    )


# ============================================================
# BUILD TRAINING PROMPT
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
# PREPARE TRAINING DATA
# ============================================================

def prepare_dataset(
    data,
    tables_lookup,
    mode
):

    records = []

    for example in data:

        db_id = example[
            "db_id"
        ]

        question = get_question(
            example,
            mode
        )

        sql = get_sql(
            example,
            mode
        )

        schema = build_schema(
            db_id,
            tables_lookup
        )

        prompt = build_prompt(
            db_id,
            schema,
            question
        )

        records.append(
            {
                "prompt": prompt,
                "sql": sql,
            }
        )

    return Dataset.from_list(
        records
    )


# ============================================================
# TOKENIZE TRAINING EXAMPLE
#
# IMPORTANT:
# Loss is calculated ONLY on SQL tokens.
# Prompt tokens are masked with -100.
# ============================================================

def tokenize_example(
    example,
    tokenizer,
    max_length
):

    prompt = example[
        "prompt"
    ]

    sql = example[
        "sql"
    ]

    # --------------------------------------------------------
    # Tokenize prompt
    # --------------------------------------------------------

    prompt_tokens = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )

    prompt_ids = prompt_tokens[
        "input_ids"
    ]

    # --------------------------------------------------------
    # Tokenize SQL
    # --------------------------------------------------------

    sql_tokens = tokenizer(
        sql,
        add_special_tokens=False,
    )

    sql_ids = sql_tokens[
        "input_ids"
    ]

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    input_ids = (
        prompt_ids +
        sql_ids
    )

    labels = (
        [-100] * len(prompt_ids)
        +
        sql_ids
    )

    attention_mask = [
        1
        for _ in input_ids
    ]

    # --------------------------------------------------------
    # Truncate complete sequence
    # --------------------------------------------------------

    input_ids = input_ids[
        :max_length
    ]

    labels = labels[
        :max_length
    ]

    attention_mask = attention_mask[
        :max_length
    ]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print()
    print("=" * 60)
    print("Loading Qwen2.5-32B-Instruct")
    print("=" * 60)

    print(
        f"Model: {MODEL_ID}"
    )

    print(
        "Quantization: 4-bit NF4"
    )

    # --------------------------------------------------------
    # 4-bit QLoRA configuration
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
    # Model
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
    # Memory-efficient QLoRA preparation
    #
    # On a ~20 GB GPU, PEFT's standard
    # prepare_model_for_kbit_training() can trigger a large
    # FP32 allocation after the 4-bit model has already filled
    # most of VRAM. Avoid that conversion and enable only the
    # pieces required for LoRA training.
    # --------------------------------------------------------

    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        }
    )
    model.enable_input_require_grads()

    # --------------------------------------------------------
    # LoRA configuration
    # --------------------------------------------------------

    lora_config = LoraConfig(

        # Conservative settings for Qwen2.5-32B on ~20 GB VRAM.
        r=8,

        lora_alpha=16,

        lora_dropout=0.05,

        bias="none",

        task_type="CAUSAL_LM",

        # Attention-only adapters use substantially less VRAM
        # than adapting all MLP projections.
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )

    # --------------------------------------------------------
    # Add LoRA adapters
    # --------------------------------------------------------

    model = get_peft_model(
        model,
        lora_config
    )

    # --------------------------------------------------------
    # Print trainable parameters
    # --------------------------------------------------------

    model.print_trainable_parameters()

    return model, tokenizer


# ============================================================
# MEMORY-EFFICIENT TRAINER
# ============================================================
#
# Qwen2.5-32B has a very large vocabulary. A normal causal-LM
# training step computes logits for EVERY token:
#
#     [batch, sequence_length, vocab_size]
#
# On a ~19.5 GB GPU this can OOM inside cross_entropy even
# though the forward pass itself fits.
#
# Qwen2 supports `logits_to_keep`, which allows us to compute
# logits only for the positions whose labels actually matter.
# Since this dataset masks the prompt with -100, we only need
# logits for the SQL prediction positions.
#
# This dramatically reduces the peak memory used by the LM head
# and cross-entropy calculation.
# ============================================================

class SQLOnlyMemoryEfficientTrainer(Trainer):

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        labels = inputs.pop("labels")

        # We intentionally use batch size 1 because logits_to_keep
        # accepts one shared set of sequence indices for the batch.
        if labels.shape[0] != 1:
            raise RuntimeError(
                "SQLOnlyMemoryEfficientTrainer requires "
                "per_device_train_batch_size=1."
            )

        # Standard causal-LM alignment:
        # logits at position t predict labels at position t+1.
        shift_labels = labels[:, 1:]

        valid_mask = shift_labels[0].ne(-100)
        valid_positions = torch.nonzero(
            valid_mask,
            as_tuple=False,
        ).flatten()

        if valid_positions.numel() == 0:
            raise RuntimeError(
                "This example contains no valid SQL target tokens."
            )

        # Qwen2's logits_to_keep accepts a 1-D tensor of sequence
        # positions. These are the positions in the ORIGINAL input
        # sequence whose logits we need.
        #
        # Example:
        # labels = [-100, -100, SQL1, SQL2, SQL3]
        # shift_labels = [-100, SQL1, SQL2, SQL3]
        # valid_positions = [1, 2, 3]
        #
        # Therefore logits at positions 1,2,3 are exactly the
        # predictions for SQL1, SQL2, SQL3.
        valid_positions = valid_positions.to(
            device=inputs["input_ids"].device,
            dtype=torch.long,
        )

        # Verify that the installed Transformers/Qwen2 actually
        # supports logits_to_keep.
        import inspect

        if "logits_to_keep" not in inspect.signature(
            model.forward
        ).parameters:
            raise RuntimeError(
                "\nYour installed Transformers version does not expose "
                "`logits_to_keep` in Qwen2ForCausalLM.forward.\n"
                "This memory-saving trainer requires a Transformers "
                "version with Qwen2 `logits_to_keep` support.\n"
                "Please upgrade Transformers before training, e.g.:\n"
                "    pip install -U transformers\n"
            )

        # Do NOT pass labels to the model. Otherwise the model's
        # standard loss path would again construct a large full-
        # sequence logits tensor.
        model_inputs = {
            key: value
            for key, value in inputs.items()
            if key != "labels"
        }

        outputs = model(
            **model_inputs,
            labels=None,
            logits_to_keep=valid_positions,
        )

        logits = outputs.logits[0]

        # The selected target labels correspond one-to-one with
        # the selected logits.
        targets = shift_labels[0, valid_mask].to(
            device=logits.device,
            dtype=torch.long,
        )

        # Calculate CE in small chunks. This avoids a second large
        # temporary allocation inside cross_entropy for a large
        # vocabulary.
        chunk_size = 32
        total_loss = None
        total_tokens = targets.numel()

        for start in range(0, total_tokens, chunk_size):
            end = min(
                start + chunk_size,
                total_tokens,
            )

            chunk_logits = logits[start:end]
            chunk_targets = targets[start:end]

            # Keep logits in bf16; do not upcast the complete
            # [SQL_tokens, vocab] matrix to float32.
            chunk_loss = F.cross_entropy(
                chunk_logits,
                chunk_targets,
                reduction="sum",
            )

            total_loss = (
                chunk_loss
                if total_loss is None
                else total_loss + chunk_loss
            )

        loss = total_loss / total_tokens

        if return_outputs:
            return loss, outputs

        return loss


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "pos",
            "neg"
        ],
        help="Train on positive or negative Spider data",
    )

    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Training JSON file",
    )

    parser.add_argument(
        "--tables_file",
        type=str,
        required=True,
        help="Spider tables.json",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for LoRA adapter",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length. 2048 is recommended for Qwen2.5-32B on ~20 GB VRAM.",
    )

    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=3.0,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size per GPU. Must remain 1 for memory-efficient logits selection",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="LoRA learning rate",
    )

    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Warmup steps",
    )

    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Logging frequency",
    )

    parser.add_argument(
        "--save_steps",
        type=int,
        default=250,
        help="Checkpoint save frequency",
    )

    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="Maximum checkpoints to keep",
    )

    args = parser.parse_args()

    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

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
    # LOAD TRAINING DATA
    # ========================================================

    print()
    print(
        f"Loading {args.mode} training data..."
    )

    train_data = load_json(
        args.train_file
    )

    print(
        f"Training examples: "
        f"{len(train_data)}"
    )

    # ========================================================
    # PREPARE DATASET
    # ========================================================

    print()
    print("Preparing dataset...")

    dataset = prepare_dataset(
        train_data,
        tables_lookup,
        args.mode
    )

    print(
        f"Dataset size: "
        f"{len(dataset)}"
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    model, tokenizer = load_model()

    print()
    print("=" * 60)
    print("MEMORY-SAFE QLORA CONFIGURATION")
    print("=" * 60)
    print("QLoRA preparation : manual (avoids FP32 PEFT conversion)")
    print("LoRA rank         : 8")
    print("LoRA alpha        : 16")
    print("LoRA targets      : q/k/v/o projections only")
    print(f"Max sequence len  : {args.max_length}")
    print("Gradient checkpointing: enabled")
    print("Memory-efficient logits : SQL positions only")
    print("CE chunk size            : 32 tokens")
    print("=" * 60)

    # ========================================================
    # TOKENIZE
    # ========================================================

    print()
    print("Tokenizing dataset...")

    tokenized_dataset = dataset.map(

        lambda example:
            tokenize_example(
                example,
                tokenizer,
                args.max_length
            ),

        remove_columns=[
            "prompt",
            "sql"
        ],

        desc="Tokenizing",
    )

    # ========================================================
    # TRAINING ARGUMENTS
    # ========================================================

    training_args = TrainingArguments(

        output_dir=args.output_dir,

        num_train_epochs=args.num_train_epochs,

        per_device_train_batch_size=(
            args.per_device_train_batch_size
        ),

        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
        ),

        learning_rate=args.learning_rate,

        warmup_steps=args.warmup_steps,

        lr_scheduler_type="cosine",

        logging_steps=args.logging_steps,

        save_strategy="steps",

        save_steps=args.save_steps,

        save_total_limit=args.save_total_limit,

        bf16=True,

        fp16=False,

        gradient_checkpointing=True,

        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },

        optim="paged_adamw_8bit",

        weight_decay=0.01,

        max_grad_norm=1.0,

        # Memory-friendly data loading.
        dataloader_num_workers=0,
        dataloader_pin_memory=False,

        report_to="none",

        remove_unused_columns=False,


        ddp_find_unused_parameters=False,
    )

    # ========================================================
    # DATA COLLATOR
    # ========================================================

    data_collator = DataCollatorForSeq2Seq(

        tokenizer=tokenizer,

        padding=True,

        label_pad_token_id=-100,

        return_tensors="pt",
    )

    # ========================================================
    # FREE UNUSED CUDA MEMORY BEFORE TRAINING
    # ========================================================

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ========================================================
    # TRAINER
    # ========================================================

    trainer = SQLOnlyMemoryEfficientTrainer(

        model=model,

        args=training_args,

        train_dataset=tokenized_dataset,

        data_collator=data_collator,
    )

    # ========================================================
    # FIND CHECKPOINT
    # ========================================================

    resume_checkpoint = None

    if os.path.exists(
        args.output_dir
    ):

        checkpoints = []

        for name in os.listdir(
            args.output_dir
        ):

            if name.startswith(
                "checkpoint-"
            ):

                full_path = os.path.join(
                    args.output_dir,
                    name
                )

                if os.path.isdir(
                    full_path
                ):

                    try:

                        step = int(
                            name.split(
                                "-"
                            )[-1]
                        )

                        checkpoints.append(
                            (
                                step,
                                full_path
                            )
                        )

                    except ValueError:

                        pass

        if checkpoints:

            checkpoints.sort(
                key=lambda x: x[0]
            )

            resume_checkpoint = (
                checkpoints[-1][1]
            )

            print()
            print(
                "Resuming from checkpoint:"
            )

            print(
                resume_checkpoint
            )

    # ========================================================
    # TRAIN
    # ========================================================

    print()
    print("=" * 60)
    print("STARTING QLORA TRAINING")
    print("=" * 60)

    trainer.train(
        resume_from_checkpoint=
            resume_checkpoint
    )

    # ========================================================
    # SAVE FINAL ADAPTER
    # ========================================================

    print()
    print("Saving final LoRA adapter...")

    trainer.save_model(
        args.output_dir
    )

    tokenizer.save_pretrained(
        args.output_dir
    )

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Adapter saved to:"
    )

    print(
        args.output_dir
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()