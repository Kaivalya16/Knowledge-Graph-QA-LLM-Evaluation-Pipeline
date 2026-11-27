# NegQA: Generating and Evaluating Negated Question-Answer Pairs

**Do LLMs understand "not"?** This project builds a validated benchmark of **positive / negated question pairs** for Text-to-SQL and uses it to evaluate and fine-tune open LLMs (Qwen2.5-7B, Llama-3.1-8B, Gemma-2-9B).

Starting from the [Spider](https://yale-lily.github.io/spider) benchmark, the pipeline automatically negates a question **and** its SQL together (for example, *"singers from France"* → *"singers not from France"*). Every generated pair is kept only if it passes **execution-based validation**: both queries are run on the real database and their answers are checked. The resulting dataset is used to compare few-shot prompting with QLoRA fine-tuning on positive and negated questions.

> **Status:** ongoing. This repository currently contains the Spider / SQL part of the project (`Spider_Evaluation_SLMs/`). The knowledge-graph part (SPARQL over Apache Jena Fuseki, ComplexWebQuestions) will be added later.

---

## Key results

Execution accuracy (%) on the 475-question test sets:

| Model | Setting | Positive questions | Negated questions |
|---|---|---|---|
| Qwen2.5-7B-Instruct | 3-shot (mean ± std, 3 shot sets) | 77.19 ± 1.58 | 73.47 ± 0.56 |
| | QLoRA fine-tuned (run 1 / run 2) | 87.58 / 85.05 | 82.32 / 83.16 |
| Llama-3.1-8B-Instruct | 3-shot | 74.81 ± 0.85 | 72.35 ± 0.95 |
| | QLoRA fine-tuned | 85.89 / 85.68 | 86.32 / 86.74 |
| Gemma-2-9B-IT | 3-shot | 75.09 ± 1.22 | 73.54 ± 0.68 |
| | QLoRA fine-tuned | 87.37 / 87.37 | 85.47 / 85.47 |

**Findings**

- **Negation is harder in few-shot mode.** Every model scores 1.5–3.7 points lower on negated questions than on positive ones.
- **Fine-tuning helps a lot.** QLoRA on about 1,100 examples raises execution accuracy by roughly 9–14 points. It mostly closes the positive/negative gap; Llama's negative score ends slightly above its positive one.
- **Exact match under-counts correct SQL.** Few-shot exact match is only 25–46% while execution accuracy is 72–77%, so many correct queries are simply written differently from the gold SQL. Execution accuracy is therefore the main metric.
- **Ignored negation is rare but real.** In few-shot mode, 1.2–1.7% of negated questions return the *positive* question's answer, meaning the model dropped the "not". After fine-tuning this falls to about 0.6–1%.
- **Remaining weak spots:** `equal → not equal` (the largest category, 176 test questions) stays at about 78–81%, and questions whose negation sits in a prepositional phrase reach only about 71–76%.

---

## Pipeline

```
Spider (train_spider, train_others, dev)
   │
   ▼  Stage 1  scripts/01_create_canonical_positive.py
   │           Sort every question into: positive / native negative / rejected
   │           (native negatives are converted into a positive partner)
   ▼  Stage 2  scripts/02_generate_negation.py
   │           Negate one value predicate of the SQL and rewrite the question
   │           so the "not" lands on the same predicate
   ▼  Stage 3  scripts/03_filter_validate.py
   │           Execute both queries; keep ONE validated pair per question
   ▼  Stage 4  scripts/04_merge_datasets.py
   │           Merge splits with integrity checks (ids, alignment, answers)
   ▼  Stage 5  scripts/05_finalize_and_split.py
   │           Remove duplicates, re-execute everything, add native pairs,
   │           70/30 train/test split
   ▼  Final    final_datasets_checked/filter_valid_pairs.py
               Strict check for scope errors, unsound negation, ambiguity
               → pos_train / pos_test / neg_train / neg_test
```

### Negation generation (Stage 2)

Each **value predicate** (a column compared with a literal or a subquery) is negated, and the question is rewritten with grammar-aware rules anchored on that predicate's literal:

```text
positive  Show the names of singers from France.
          SELECT name FROM singer WHERE country = 'France'
negated   Show the names of singers not from France.
          SELECT name FROM singer WHERE country != 'France'
```

Design rules that keep the pairs sound:

- JOIN keys (`T1.id = T2.id`) are never negated.
- A two-part name is negated as one entity: `NOT (first_name = 'Linda' AND last_name = 'Smith')`.
- With coordinated conditions (and / or / but), only the last condition is negated.
- Every rewritten question must contain exactly one more negation word than the positive one and must not match known ungrammatical patterns.

Supported negation types: `=`→`!=`, `>`→`not >`, `<`→`not <`, `>=`, `<=`, `BETWEEN`, `LIKE`, `IN`, and multi-column entities.

### Execution-based validation (Stages 3 and 5)

A pair is rejected if any of these hold:

| Check | Why it matters |
|---|---|
| Positive or negative answer is empty | An empty answer is not a useful test question |
| Negative answer contains only NULLs | For example, `AVG` over no rows |
| Negative answer has more than 10 rows | Keeps answers checkable |
| Negative answer equals positive answer | The negation changed nothing |
| Stored answer differs on re-execution | Guards against stale or wrong gold answers |
| Duplicate pair | Spider repeats some questions verbatim |

The final strict filter also drops pairs with **negation-scope errors**, **unsound negation** (such as vacuous `OR NOT`), **ambiguous scope**, and **comparator mismatches** between question and SQL. Every dropped pair is logged with its reason in `dropped.csv`.

### Dataset funnel

| Stage | Questions / pairs |
|---|---|
| Original Spider (train_spider + train_others + dev) | 9,693 |
| Canonical positives | 7,109 |
| Questions with at least one negation candidate | 2,587 |
| Validated generated pairs (one per question) | 1,643 |
| **Final strict pairs** (generated + native) | **1,582** (1,190 generated, 392 native) |
| Train / test (70 / 30) | 1,107 / 475 |

*Native* pairs come from questions Spider's annotators already wrote in negated form; each is paired with an automatically de-negated positive so both sets stay the same size. The `train_others` databases in the Spider release are empty, so they contribute no pairs.

---

## Models and training

**Models:** `Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`, `google/gemma-2-9b-it`.

**Prompt:** all scripts share `text2sql_prompt.py`, so training, inference and few-shot use identical prompts. The prompt contains the rules (including rules on negation scope), the database schema with column types, primary keys and foreign keys, and the question. The model outputs SQL only.

**Few-shot baseline:** 3 demonstrations chosen by TF-IDF similarity to the test question (with a bonus for the same database), greedy decoding. The run is repeated with 3 different demonstration sets and reported as mean ± std. Prompts are never truncated; the least similar demonstrations are dropped until the prompt fits.

**QLoRA fine-tuning:**

| Setting | Value |
|---|---|
| Base model precision | 4-bit NF4, double quantization, bf16 compute |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Learning rate | 1e-4, cosine schedule, 5% warmup |
| Effective batch size | 16 (1 × 16 gradient accumulation) |
| Epochs | 3 |
| Optimizer | paged AdamW 8-bit, gradient checkpointing |
| Loss | On SQL tokens only (prompt tokens masked with -100) |

Separate adapters are trained for positive and negated questions. Implementation details worth noting:

- Examples longer than `max_length` are **skipped, never truncated**, so the model never learns to write SQL for a schema it could not see.
- The prompt/answer token boundary is verified for every example.
- **Gemma memory fix:** Gemma's 256K-token vocabulary makes the standard loss build a huge float32 logits matrix at every position, which ran out of memory on a 20 GB GPU for long schemas. A custom trainer computes logits only at the SQL positions, giving the same loss and gradients with far less memory.
- `--debug_overfit` trains on 20 examples for 50 epochs as a sanity check that the pipeline can learn.

**Evaluation:** every prediction is executed on the Spider SQLite database and compared with the gold answer. Results are broken down by negation type, by question-rewrite rule and by source split. With `--paired_file`, the script also counts predictions that return the *paired* question's answer.

---

## Repository structure

```
Spider_Evaluation_SLMs/
├── scripts/                     # dataset construction pipeline (Stages 1–5)
│   ├── 01_create_canonical_positive.py
│   ├── 02_generate_negation.py
│   ├── 03_filter_validate.py
│   ├── 04_merge_datasets.py
│   ├── 05_finalize_and_split.py
│   └── split_dataset.py, json_to_csv.py, count_examples.py   # utilities
├── datasets/
│   ├── native_negatives_all.json
│   └── stage4_report.json       # funnel and negation-type statistics
├── final_datasets_checked/      # final benchmark
│   ├── pos_train.json  pos_test.json  neg_train.json  neg_test.json
│   ├── positive_strict.json  negative_strict.json
│   ├── positive_valid.json   negative_valid.json
│   ├── dropped.csv              # every removed pair with its reason
│   └── filter_valid_pairs.py
├── fewshot/                     # few-shot baselines + prediction/summary files
│   ├── 01_qwen.py  02_llama.py  03_gemma.py
│   └── text2sql_prompt.py
└── finetune/                    # QLoRA training, inference + result files
    ├── 01_finetune_qwen.py   02_inference_qwen.py
    ├── 03_finetune_llama.py  04_inference_llama.py
    ├── 05_finetune_gemma.py  06_inference_gemma.py
    └── text2sql_prompt.py
```

Each record in the final datasets looks like this:

```json
{
  "id": "train_spider_866",
  "db_id": "chinook_1",
  "source_type": "generated",
  "question": "Count the number of tracks that are of the media type that is not \"AAC audio file\".",
  "sql": "SELECT COUNT(*) FROM MEDIATYPE AS T1 JOIN TRACK AS T2 ON T1.MediaTypeId = T2.MediaTypeId WHERE T1.Name != \"AAC audio file\"",
  "answer": [["3492"]],
  "negation_type": "equal_to_not_equal",
  "question_rule": "contrast"
}
```

---

## Getting started

### Requirements

```bash
pip install torch transformers peft bitsandbytes accelerate datasets numpy tqdm
pip install lemminflect   # optional: better verb forms when rewriting questions
```

A CUDA GPU with about 20 GB of memory is enough for QLoRA on all three models. Llama 3.1 and Gemma 2 are gated on Hugging Face: accept their licences and run `huggingface-cli login`.

Download Spider and place `tables.json` and the `database/` folder under `data/raw/`.

### Fine-tune and evaluate

```bash
cd Spider_Evaluation_SLMs/finetune

# train a negation adapter
python 01_finetune_qwen.py --mode neg \
    --input_file ../final_datasets_checked/neg_train.json \
    --tables_file data/raw/tables.json \
    --output_dir adapters/qwen_neg

# evaluate it with execution accuracy and the paired-answer check
python 02_inference_qwen.py --mode neg \
    --input_file ../final_datasets_checked/neg_test.json \
    --paired_file ../final_datasets_checked/pos_test.json \
    --tables_file data/raw/tables.json \
    --database_dir data/raw/database \
    --adapter_dir adapters/qwen_neg \
    --output_file results/qwen_neg_on_neg.jsonl
```

Omit `--adapter_dir` to evaluate the zero-shot base model. Use `03_/04_` for Llama and `05_/06_` for Gemma.

### Few-shot baseline

```bash
cd Spider_Evaluation_SLMs/fewshot
python 01_qwen.py --mode neg \
    --train_file ../final_datasets_checked/neg_train.json \
    --test_file ../final_datasets_checked/neg_test.json \
    --paired_file ../final_datasets_checked/pos_test.json \
    --tables_file data/raw/tables.json \
    --database_dir data/raw/database \
    --num_shots 3 --num_iterations 3 \
    --output_file results/fewshot_qwen_neg.jsonl
```

---

## Limitations and next steps

- Paraphrases of the same SQL are kept as independent pairs, so two paraphrases can fall on different sides of the train/test split. A split by SQL group would give a stricter test of generalisation.
- Each test set has 475 questions, and some negation types have fewer than 10 examples, so per-type accuracies there are noisy.
- No LoRA rank or learning-rate sweep has been run yet.
- **Next:** extend the pipeline to knowledge-graph QA (SPARQL on Apache Jena Fuseki, ComplexWebQuestions), where the open-world assumption makes negation harder: a missing triple does not mean the fact is false.

## Acknowledgements

Built on the [Spider](https://yale-lily.github.io/spider) dataset (Yu et al., 2018). Fine-tuning uses Hugging Face Transformers, PEFT and bitsandbytes (QLoRA, Dettmers et al., 2023).
