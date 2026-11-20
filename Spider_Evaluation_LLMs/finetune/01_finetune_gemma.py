#!/usr/bin/env python3
"""
Gemma-3-27B-IT QLoRA fine-tuning for Spider positive / negative Text-to-SQL.

Same pipeline, prompt content (text2sql_prompt.py), data handling, options
and SQL-only loss as 05_finetune_gemma.py (Gemma 2). What changes for Gemma 3:

- google/gemma-3-27b-it is gated: accept the licence on its Hugging Face
  page and log in once (`huggingface-cli login`). Needs transformers >= 4.50.
- The 27B checkpoint is multimodal (Gemma3ForConditionalGeneration: text
  decoder + SigLIP vision tower). It is loaded with that class, because
  Gemma3ForCausalLM would not find the decoder weights under their
  "language_model" prefix. Only text is used; LoRA is restricted to the
  text decoder (the vision tower also has q/k/v projections).
- The rules still open the first user turn. Gemma 3's template does accept
  a system message, but it only does the same merge; doing it here keeps
  the text identical to the Gemma 2 runs and independent of the template
  version.
- No attention soft-capping in Gemma 3 (it uses QK-norm), so "sdpa" works
  for training and is the default (much less memory than "eager" at 4k
  tokens). --attn_implementation eager is still available.
- Older transformers (4.50-4.5x) require token_type_ids when the multimodal
  model is trained through its own forward (--full_logits_loss); every row
  carries all-zero token_type_ids (text only), which is harmless elsewhere.
- Memory: the vocabulary has 262,144 tokens and the embedding/output matrix
  (262k x 5376) is ~2.8 GB in bfloat16. SqlOnlyLossTrainer builds logits only
  at the SQL positions, and the matrix is not up-cast to float32 (+2.8 GB).
  The 4-bit 27B model needs roughly 18 GB for weights; with gradient
  checkpointing a 40-48 GB GPU is comfortable for 4k-token examples.
- Defaults: LoRA r=16 / alpha=32 on q,k,v,o,gate,up,down of the decoder,
  batch 1 x 16 accumulation, LR 1e-4 - unchanged, so runs stay comparable.

Example:
  python 07_finetune_gemma3.py --mode pos \\
      --input_file final_datasets/pos_train.json \\
      --tables_file data/raw/tables.json --output_dir adapters/gemma3_pos
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import inspect
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from datasets import Dataset
from packaging import version
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text2sql_prompt import build_messages, get_sql, load_schema_map, normalize_sql  # noqa: E402


MODEL_NAME = "google/gemma-3-27b-it"

END_OF_TURN = "<end_of_turn>"

# LoRA on the text decoder only: any projection outside the vision tower
# and the multimodal projector (module names differ across transformers
# versions, so a pattern is used instead of a fixed prefix).
LORA_TARGET_REGEX = (r"^(?!.*(vision_tower|multi_modal_projector)).*\."
                     r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$")


# ------------------------------------------------------------
# ARGUMENTS
# ------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["pos", "neg", "mix"], required=True,
                        help="Label for the run; 'mix' expects several input files.")
    parser.add_argument("--input_file", nargs="+", required=True,
                        help="One or more training JSON files.")
    parser.add_argument("--tables_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--attn_implementation", default="sdpa",
                        choices=["sdpa", "eager", "flash_attention_2"])

    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="If > 0, overrides --warmup_ratio.")
    parser.add_argument("--max_length", type=int, default=4096)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--val_ratio", type=float, default=0.0,
                        help="Hold out this share of the TRAIN data (by paraphrase "
                             "group) for eval loss. 0 = no validation.")
    parser.add_argument("--no_schema_keys", action="store_true",
                        help="Schema without types/keys (v3 format).")

    parser.add_argument("--full_logits_loss", action="store_true",
                        help="Standard Trainer loss (logits at every position; "
                             "much more memory).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Use only the first N examples.")
    parser.add_argument("--debug_overfit", action="store_true",
                        help="20 samples, 50 epochs, LR 2e-4, grad accumulation 1.")

    args = parser.parse_args()
    if args.mode != "mix" and len(args.input_file) > 1:
        parser.error("Several --input_file values need --mode mix.")
    return args


args = parse_args()

if args.debug_overfit:
    args.max_train_samples = 20
    args.num_train_epochs = 50
    args.learning_rate = 2e-4
    args.gradient_accumulation_steps = 1
    args.val_ratio = 0.0

if version.parse(transformers.__version__) < version.parse("4.50.0"):
    raise RuntimeError(f"Gemma 3 needs transformers >= 4.50 (found {transformers.__version__}).")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(args.seed)

print("=" * 80)
print("Gemma-3-27B-IT QLoRA Fine-tuning")
print("=" * 80)
for key, value in vars(args).items():
    print(f"{key:<28}: {value}")
print(f"{'transformers':<28}: {transformers.__version__}")
print("=" * 80)


# ------------------------------------------------------------
# TOKENIZER AND SCHEMA
# ------------------------------------------------------------

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
if not getattr(tokenizer, "chat_template", None):
    # some versions keep the template only on the processor
    from transformers import AutoProcessor
    tokenizer.chat_template = AutoProcessor.from_pretrained(args.model_name).chat_template
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading Spider tables...")
schema_map = load_schema_map(
    args.tables_file,
    with_types=not args.no_schema_keys,
    with_keys=not args.no_schema_keys,
)
print(f"Loaded {len(schema_map)} databases.")


def gemma_messages(messages):
    """Put the rules at the top of the first user turn (what Gemma 3's
    template would do with a system message, and what Gemma 2 required)."""
    if messages and messages[0]["role"] == "system":
        system, rest = messages[0]["content"], [dict(m) for m in messages[1:]]
        rest[0]["content"] = system + "\n\n" + rest[0]["content"]
        return rest
    return messages


def prompt_text(example):
    return tokenizer.apply_chat_template(
        gemma_messages(build_messages(example, schema_map, with_answer=False)),
        tokenize=False, add_generation_prompt=True,
    )


def full_text(example):
    return tokenizer.apply_chat_template(
        gemma_messages(build_messages(example, schema_map, with_answer=True)),
        tokenize=False, add_generation_prompt=False,
    )


def answer_text(example):
    return normalize_sql(get_sql(example)) + END_OF_TURN


# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

print("\nLoading training JSON...")
raw_data = []
for path in args.input_file:
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    for row in rows:
        row = dict(row)
        row["_source_file"] = os.path.basename(path)
        raw_data.append(row)
    print(f"  {path}: {len(rows)} examples")

if args.max_train_samples is not None:
    raw_data = raw_data[:args.max_train_samples]
print(f"Input examples: {len(raw_data)}")


def split_validation(rows, ratio, seed):
    """Hold out whole paraphrase groups so no question leaks into training."""
    if ratio <= 0:
        return rows, []
    groups = sorted({r.get("paraphrase_group") or r.get("id") for r in rows})
    random.Random(seed).shuffle(groups)
    held = set(groups[:max(1, round(ratio * len(groups)))])
    train = [r for r in rows if (r.get("paraphrase_group") or r.get("id")) not in held]
    val = [r for r in rows if (r.get("paraphrase_group") or r.get("id")) in held]
    return train, val


train_rows, val_rows = split_validation(raw_data, args.val_ratio, args.seed)
print(f"Train examples: {len(train_rows)}   validation examples: {len(val_rows)}")


# ------------------------------------------------------------
# TOKENIZATION
# ------------------------------------------------------------

def tokenize_rows(rows, label):
    tokenized, lengths, prompt_lengths, sql_lengths = [], [], [], []
    skipped_long, skipped_bad = [], []

    for index, example in enumerate(rows):
        try:
            prompt = prompt_text(example)
            answer = answer_text(example)
            # The chat template must render prompt + SQL + <end_of_turn> + "\n";
            # the final newline is not trained.
            if full_text(example).rstrip("\n") != prompt + answer:
                raise ValueError("Chat template does not end the turn with SQL + <end_of_turn>.")

            # The template text already starts with <bos>, so no special
            # tokens are added (that would give a second BOS).
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
            full_ids = prompt_ids + answer_ids

            prompt_len = len(prompt_ids)
            sql_len = len(answer_ids)
            if sql_len <= 0:
                raise ValueError("No assistant SQL tokens detected.")

            lengths.append(len(full_ids))
            prompt_lengths.append(prompt_len)
            sql_lengths.append(sql_len)

            # Never keep only part of an oversized schema: skip instead.
            if len(full_ids) > args.max_length:
                skipped_long.append({"index": index, "id": example.get("id"),
                                     "total_tokens": len(full_ids),
                                     "prompt_tokens": prompt_len, "sql_tokens": sql_len})
                continue

            tokenized.append({
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "token_type_ids": [0] * len(full_ids),  # 0 = text (no image tokens)
                "labels": [-100] * prompt_len + full_ids[prompt_len:],
            })
        except Exception as exc:
            skipped_bad.append({"index": index, "id": example.get("id"), "error": str(exc)})

    print("\n" + "=" * 80)
    print(f"{label.upper()} DATA DIAGNOSTICS")
    print("=" * 80)
    print(f"Examples seen           : {len(rows)}")
    print(f"Examples used           : {len(tokenized)}")
    print(f"Skipped > max_length    : {len(skipped_long)}")
    print(f"Skipped due to errors   : {len(skipped_bad)}")
    if skipped_bad:
        print(f"  first error           : {skipped_bad[0]['error']}")
    if lengths:
        print(f"Total tokens min/avg/max: {min(lengths)} / {np.mean(lengths):.1f} / {max(lengths)}")
        print(f"Average prompt tokens   : {np.mean(prompt_lengths):.1f}")
        print(f"Average SQL tokens      : {np.mean(sql_lengths):.1f}")
        for limit in (512, 1024, 2048, 4096):
            count = sum(x > limit for x in lengths)
            print(f"Examples > {limit:4d} tokens: {count}/{len(lengths)} "
                  f"({100.0 * count / len(lengths):.2f}%)")
    print("=" * 80)
    return tokenized, skipped_long, skipped_bad


train_tokenized, skipped_long, skipped_bad = tokenize_rows(train_rows, "training")
if not train_tokenized:
    raise RuntimeError("No usable training examples remain - see the errors above.")
val_tokenized = tokenize_rows(val_rows, "validation")[0] if val_rows else []

sample = train_rows[0]
print("\nTRAINING SAMPLE")
print("=" * 80)
print(full_text(sample))
print(f"Target SQL: {normalize_sql(get_sql(sample))}")
print("=" * 80)

steps_per_epoch = max(1, len(train_tokenized) //
                      (args.per_device_train_batch_size * args.gradient_accumulation_steps))
total_steps = int(steps_per_epoch * args.num_train_epochs)
warmup = args.warmup_steps if args.warmup_steps > 0 else int(args.warmup_ratio * total_steps)
print(f"\nOptimizer steps: ~{total_steps} ({steps_per_epoch}/epoch), warmup steps: {warmup}")

train_dataset = Dataset.from_list(train_tokenized)
val_dataset = Dataset.from_list(val_tokenized) if val_tokenized else None


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

def model_class(name):
    """The multimodal 4B/12B/27B checkpoints need Gemma3ForConditionalGeneration;
    a text-only checkpoint (e.g. gemma-3-1b-it) loads as a causal LM."""
    if AutoConfig.from_pretrained(name).model_type == "gemma3":
        from transformers import Gemma3ForConditionalGeneration
        return Gemma3ForConditionalGeneration
    return AutoModelForCausalLM


def dtype_kwargs():
    # `torch_dtype` was renamed `dtype` in transformers 4.56
    key = "dtype" if version.parse(transformers.__version__) >= version.parse("4.56.0") else "torch_dtype"
    return {key: torch.bfloat16}


print("\nLoading model in 4-bit NF4...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
ModelClass = model_class(args.model_name)
print(f"Model class: {ModelClass.__name__}")
model = ModelClass.from_pretrained(
    args.model_name,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=args.attn_implementation,
    trust_remote_code=True,
    **dtype_kwargs(),
)
model.config.use_cache = False
if hasattr(model.config, "text_config"):
    model.config.text_config.use_cache = False

# prepare_model_for_kbit_training would also up-cast every non-quantized
# weight to float32, including the 262k x 5376 embedding / output matrix
# (+2.8 GB), so only its gradient-checkpointing part is done here.
for parameter in model.parameters():
    parameter.requires_grad = False
try:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
except TypeError:  # older transformers
    model.gradient_checkpointing_enable()
model.enable_input_require_grads()

lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=LORA_TARGET_REGEX,
)
model = get_peft_model(model, lora_config)
print("\nLoRA trainable parameters:")
model.print_trainable_parameters()

lora_modules = [n for n, m in model.named_modules() if hasattr(m, "lora_A")]
if any("vision" in n for n in lora_modules):
    raise RuntimeError("LoRA reached the vision tower - check LORA_TARGET_REGEX.")
print(f"LoRA modules: {len(lora_modules)} (e.g. {lora_modules[0] if lora_modules else '-'})")


# ------------------------------------------------------------
# TRAINING ARGUMENTS (version tolerant)
# ------------------------------------------------------------

def make_training_arguments(**kwargs):
    supported = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in kwargs and "eval_strategy" not in supported:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    dropped = [k for k in kwargs if k not in supported]
    for key in dropped:
        kwargs.pop(key)
    if dropped:
        print(f"Note: this Transformers version ignores {dropped}")
    return TrainingArguments(**kwargs)


training_args = make_training_arguments(
    output_dir=args.output_dir,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    lr_scheduler_type="cosine",
    warmup_steps=warmup,
    weight_decay=0.01,
    logging_strategy="steps",
    logging_steps=1 if args.debug_overfit else 5,
    eval_strategy="epoch" if val_dataset is not None else "no",
    save_strategy="epoch",
    save_total_limit=2,
    bf16=True,   # Gemma 3 overflows in float16
    fp16=False,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    remove_unused_columns=False,
    prediction_loss_only=True,
    report_to="none",
    seed=args.seed,
    data_seed=args.seed,
    dataloader_num_workers=0,
    ddp_find_unused_parameters=False,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,
    pad_to_multiple_of=8,
)


def text_decoder_and_head(model):
    """Gemma 3's text decoder and output layer, across transformers versions:
    >= 4.52: Gemma3ForConditionalGeneration.model.language_model (Gemma3TextModel)
    4.50-4.51: Gemma3ForConditionalGeneration.language_model.model
    text-only: Gemma3ForCausalLM.model"""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    inner = getattr(base, "model", None)
    if inner is not None and hasattr(inner, "language_model"):
        decoder = inner.language_model
    elif hasattr(base, "language_model"):
        lm = base.language_model
        decoder = lm.model if hasattr(lm, "lm_head") else lm
    else:
        decoder = inner
    config = getattr(base.config, "text_config", base.config)
    return decoder, base.get_output_embeddings(), config


class SqlOnlyLossTrainer(Trainer):
    """Cross-entropy on the SQL tokens only, computing logits only there.

    Same loss as the model's own (final logit soft-capping is applied if the
    config sets it; Gemma 3 does not), but the 262k-wide logits are built for
    the labelled positions only. The text decoder is called directly: no
    image tokens, so the multimodal wrapper adds nothing. Each micro-batch
    returns its mean token loss; the Trainer averages over accumulation steps.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        # always let the Trainer divide by gradient_accumulation_steps
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        decoder, lm_head, config = text_decoder_and_head(model)
        hidden = decoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
        )[0]
        target = labels[:, 1:]
        mask = target != -100
        selected = hidden[:, :-1, :][mask]
        logits = lm_head(selected).float()
        cap = getattr(config, "final_logit_softcapping", None)
        if cap:
            logits = torch.tanh(logits / cap) * cap
        loss = torch.nn.functional.cross_entropy(logits, target[mask])
        return (loss, {"loss": loss}) if return_outputs else loss


TrainerClass = Trainer if args.full_logits_loss else SqlOnlyLossTrainer
print(f"\nLoss: {'standard (all logits)' if args.full_logits_loss else 'SQL positions only'}")

trainer = TrainerClass(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)


# ------------------------------------------------------------
# TRAIN AND SAVE
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("STARTING TRAINING")
print("=" * 80)
train_result = trainer.train()
print(f"\nTraining loss: {train_result.training_loss:.4f}")

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
trainer.save_model(str(output_dir))
tokenizer.save_pretrained(str(output_dir))

config = dict(vars(args), model_name=args.model_name, warmup_steps_used=warmup,
              optimizer_steps=total_steps, train_examples=len(train_tokenized),
              val_examples=len(val_tokenized), transformers_version=transformers.__version__)
files = {
    "training_config.json": config,
    "loss_history.json": trainer.state.log_history,
    "skipped_long_examples.json": skipped_long,
    "skipped_bad_examples.json": skipped_bad,
    "validation_ids.json": [r.get("id") for r in val_rows],
}
for name, content in files.items():
    with open(output_dir / name, "w", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2)

print(f"\nAdapter saved to: {output_dir}")
print("Done.")