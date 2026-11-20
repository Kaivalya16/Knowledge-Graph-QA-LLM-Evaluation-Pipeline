import argparse
import json


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):

    # Accept either a JSON array or a JSONL file.
    with open(path, "r", encoding="utf-8") as f:

        try:
            return json.load(f)

        except json.JSONDecodeError:
            pass

    return load_jsonl(path)


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(path):

    data = []

    with open(path, "r", encoding="utf-8") as f:

        for line_num, line in enumerate(f, 1):

            line = line.strip()

            if not line:
                continue

            try:
                data.append(json.loads(line))

            except json.JSONDecodeError as e:

                print(
                    f"Warning: Invalid JSON on line "
                    f"{line_num}: {e}"
                )

    return data


# ============================================================
# GET GOLD ANSWER
# ============================================================

def get_gold_answer(example):

    if "gold_answer" in example:
        return example["gold_answer"]

    if "answer" in example:
        return example["answer"]

    if "answers" in example:
        return example["answers"]

    raise KeyError(
        "Gold answer field not found. "
        f"Available fields: {list(example.keys())}"
    )


# ============================================================
# NORMALIZE CELL
# ============================================================

def normalize_cell(value):

    if value is None:
        return "null"

    return str(value).strip().lower()


# ============================================================
# NORMALIZE ANSWER
#
# Row ordering is preserved.
# Column ordering is preserved.
# ============================================================

def normalize_answer(answer):

    if answer is None:
        return []

    normalized = []

    for row in answer:

        if isinstance(row, (list, tuple)):

            normalized.append(
                [
                    normalize_cell(value)
                    for value in row
                ]
            )

        else:

            normalized.append(
                [
                    normalize_cell(row)
                ]
            )

    return normalized


# ============================================================
# EXACT MATCH
# ============================================================

def exact_match(
    gold_answer,
    predicted_answer
):

    gold = normalize_answer(
        gold_answer
    )

    pred = normalize_answer(
        predicted_answer
    )

    # Exact comparison:
    # - row order matters
    # - column order matters
    # - duplicate rows matter

    return int(
        gold == pred
    )


# ============================================================
# PARTIAL ROW MATCH
#
# Ordering is preserved.
# ============================================================

def partial_row_match(
    predicted_row,
    gold_row
):

    if predicted_row == gold_row:
        return True

    # Prefix matching
    if len(predicted_row) <= len(gold_row):

        return (
            predicted_row ==
            gold_row[
                :len(predicted_row)
            ]
        )

    return (
        gold_row ==
        predicted_row[
            :len(gold_row)
        ]
    )


# ============================================================
# EVALUATE ONE QUESTION
# ============================================================

def evaluate_prediction(
    gold_answer,
    predicted_answer
):

    gold = normalize_answer(
        gold_answer
    )

    pred = normalize_answer(
        predicted_answer
    )


    # ========================================================
    # EM
    # ========================================================

    em_scores = []

    matched_gold_em = set()


    for p in pred:

        matched = False

        for g_idx, g in enumerate(gold):

            if g_idx in matched_gold_em:
                continue

            if p == g:

                matched = True

                matched_gold_em.add(
                    g_idx
                )

                break

        em_scores.append(
            int(matched)
        )


    # ========================================================
    # PEM
    # ========================================================

    pem_scores = []

    matched_gold_pem = set()


    for p in pred:

        matched = False

        for g_idx, g in enumerate(gold):

            if g_idx in matched_gold_pem:
                continue

            if partial_row_match(
                p,
                g
            ):

                matched = True

                matched_gold_pem.add(
                    g_idx
                )

                break

        pem_scores.append(
            int(matched)
        )


    # ========================================================
    # HIT
    # ========================================================

    hit_em = int(
        1 in em_scores
    )

    hit_pem = int(
        1 in pem_scores
    )


    # ========================================================
    # COUNTS
    # ========================================================

    num_pred = len(pred)

    num_gold = len(gold)


    # ========================================================
    # EM AVERAGE
    # ========================================================

    em_avg = (

        sum(em_scores) /
        num_pred

        if num_pred > 0

        else 0.0
    )


    # ========================================================
    # PEM AVERAGE
    # ========================================================

    pem_avg = (

        sum(pem_scores) /
        num_pred

        if num_pred > 0

        else 0.0
    )


    # ========================================================
    # EM PRECISION
    # ========================================================

    precision_em = (

        sum(em_scores) /
        num_pred

        if num_pred > 0

        else 0.0
    )


    # ========================================================
    # EM RECALL
    # ========================================================

    recall_em = (

        len(matched_gold_em) /
        num_gold

        if num_gold > 0

        else 0.0
    )


    # ========================================================
    # EM F1
    # ========================================================

    if (
        precision_em +
        recall_em
    ) > 0:

        f1_em = (
            2 *
            precision_em *
            recall_em
            /
            (
                precision_em +
                recall_em
            )
        )

    else:

        f1_em = 0.0


    # ========================================================
    # EM ACCURACY
    # ========================================================

    tp_em = len(
        matched_gold_em
    )

    fp_em = (
        num_pred -
        tp_em
    )

    fn_em = (
        num_gold -
        tp_em
    )

    total_em = (
        tp_em +
        fp_em +
        fn_em
    )

    accuracy_em = (

        tp_em /
        total_em

        if total_em > 0

        else 0.0
    )


    # ========================================================
    # PEM PRECISION
    # ========================================================

    precision_pem = (

        sum(pem_scores) /
        num_pred

        if num_pred > 0

        else 0.0
    )


    # ========================================================
    # PEM RECALL
    # ========================================================

    recall_pem = (

        len(matched_gold_pem) /
        num_gold

        if num_gold > 0

        else 0.0
    )


    # ========================================================
    # PEM F1
    # ========================================================

    if (
        precision_pem +
        recall_pem
    ) > 0:

        f1_pem = (
            2 *
            precision_pem *
            recall_pem
            /
            (
                precision_pem +
                recall_pem
            )
        )

    else:

        f1_pem = 0.0


    # ========================================================
    # PEM ACCURACY
    # ========================================================

    tp_pem = len(
        matched_gold_pem
    )

    fp_pem = (
        num_pred -
        tp_pem
    )

    fn_pem = (
        num_gold -
        tp_pem
    )

    total_pem = (
        tp_pem +
        fp_pem +
        fn_pem
    )

    accuracy_pem = (

        tp_pem /
        total_pem

        if total_pem > 0

        else 0.0
    )


    return {

        "EM_scores_per_answer":
            em_scores,

        "PEM_scores_per_answer":
            pem_scores,

        "HIT_EM":
            hit_em,

        "HIT_PEM":
            hit_pem,

        "EM_avg":
            em_avg,

        "PEM_avg":
            pem_avg,

        "Precision_EM":
            precision_em,

        "Recall_EM":
            recall_em,

        "F1_EM":
            f1_em,

        "Accuracy_EM":
            accuracy_em,

        "Precision_PEM":
            precision_pem,

        "Recall_PEM":
            recall_pem,

        "F1_PEM":
            f1_pem,

        "Accuracy_PEM":
            accuracy_pem
    }


# ============================================================
# CREATE GOLD LOOKUP
# ============================================================

def build_gold_lookup(gold_data):

    lookup = {}

    for idx, example in enumerate(gold_data):

        # ----------------------------------------------------
        # Primary key: id
        # ----------------------------------------------------

        if "id" in example:

            key = (
                "id",
                str(example["id"])
            )

            lookup[key] = example


        # ----------------------------------------------------
        # Next key: index
        # ----------------------------------------------------

        if "index" in example:

            key = (
                "index",
                str(example["index"])
            )

            lookup[key] = example


        # ----------------------------------------------------
        # Fallback key: db_id + question
        # ----------------------------------------------------

        if (
            "db_id" in example
            and "question" in example
        ):

            key = (
                "question",
                example["db_id"],
                example["question"]
            )

            lookup[key] = example


        # ----------------------------------------------------
        # Also store positional index as fallback
        # ----------------------------------------------------

        lookup[
            ("position", idx)
        ] = example


    return lookup


# ============================================================
# FIND GOLD EXAMPLE
# ============================================================

def find_gold_example(
    prediction,
    gold_lookup,
    gold_data
):

    # --------------------------------------------------------
    # 0. Match by id
    # --------------------------------------------------------

    if "id" in prediction:

        key = (
            "id",
            str(prediction["id"])
        )

        if key in gold_lookup:

            return gold_lookup[key]


    # --------------------------------------------------------
    # 1. Match by index
    # --------------------------------------------------------

    if "index" in prediction:

        key = (
            "index",
            str(prediction["index"])
        )

        if key in gold_lookup:

            return gold_lookup[key]


    # --------------------------------------------------------
    # 2. Match by db_id + question
    # --------------------------------------------------------

    if (
        "db_id" in prediction
        and "question" in prediction
    ):

        key = (
            "question",
            prediction["db_id"],
            prediction["question"]
        )

        if key in gold_lookup:

            return gold_lookup[key]


    return None


# ============================================================
# MAIN EVALUATION
# ============================================================

def run_evaluation(
    predictions_file,
    gold_file,
    output_file
):

    print(
        "Loading predicted answers..."
    )

    predictions = load_jsonl(
        predictions_file
    )


    # --------------------------------------------------------
    # Every prediction must carry the executed answer
    # (output of 01_sql_to_answers.py).
    # --------------------------------------------------------

    if predictions and not any(
        "predicted_answer" in p for p in predictions
    ):

        raise KeyError(
            "No 'predicted_answer' field found. Run "
            "01_sql_to_answers.py on the predictions first."
        )


    # --------------------------------------------------------
    # Gold answers: separate file, or the `answer` field
    # stored in each prediction row.
    # --------------------------------------------------------

    if gold_file:

        print(
            "Loading gold test data..."
        )

        gold_data = load_json(
            gold_file
        )

    else:

        print(
            "No --gold_file given: using the gold "
            "answer stored in each prediction row."
        )

        gold_data = predictions


    print(
        f"Predictions available: "
        f"{len(predictions)}"
    )

    print(
        f"Gold test examples:     "
        f"{len(gold_data)}"
    )


    # ========================================================
    # BUILD GOLD LOOKUP
    # ========================================================

    gold_lookup = (
        build_gold_lookup(gold_data)
        if gold_file
        else {}
    )


    evaluated = []

    unmatched = 0


    # ========================================================
    # EVALUATE ONLY PREDICTIONS PRESENT
    # ========================================================

    total = len(
        predictions
    )


    for idx, prediction in enumerate(
        predictions
    ):

        # ----------------------------------------------------
        # Find corresponding gold example
        # ----------------------------------------------------

        if gold_file:

            gold_example = find_gold_example(

                prediction,

                gold_lookup,

                gold_data
            )

        else:

            gold_example = prediction


        # ----------------------------------------------------
        # No matching gold example
        # ----------------------------------------------------

        if gold_example is None:

            unmatched += 1

            continue


        # ----------------------------------------------------
        # Get answers
        # ----------------------------------------------------

        gold_answer = get_gold_answer(
            gold_example
        )

        predicted_answer = prediction.get(
            "predicted_answer"
        )


        # ----------------------------------------------------
        # Calculate metrics
        # ----------------------------------------------------

        metrics = evaluate_prediction(

            gold_answer,

            predicted_answer
        )


        # ----------------------------------------------------
        # Create evaluated record
        # ----------------------------------------------------

        record = dict(
            prediction
        )


        record["gold_answer"] = (
            gold_answer
        )


        record.update(
            metrics
        )


        evaluated.append(
            record
        )


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        current = idx + 1

        percent = (

            current /
            total
        ) * 100


        print(
            f"\rEvaluating: "
            f"{percent:6.2f}% "
            f"({current}/{total})",
            end="",
            flush=True
        )


    print("\n")


    # ========================================================
    # SAVE DETAILED RESULTS
    # ========================================================

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            evaluated,
            f,
            indent=2,
            ensure_ascii=False
        )


    # ========================================================
    # DATASET SUMMARY
    #
    # IMPORTANT:
    # denominator = number of predictions evaluated
    # NOT total gold test examples.
    # ========================================================

    total_q = len(
        evaluated
    )


    if total_q == 0:

        print(
            "No matching predictions "
            "were found."
        )

        return


    # --------------------------------------------------------
    # HIT
    # --------------------------------------------------------

    total_hit_em = sum(
        x["HIT_EM"]
        for x in evaluated
    )

    total_hit_pem = sum(
        x["HIT_PEM"]
        for x in evaluated
    )


    mean_hit_em = (
        total_hit_em /
        total_q
    )

    mean_hit_pem = (
        total_hit_pem /
        total_q
    )


    # --------------------------------------------------------
    # EM
    # --------------------------------------------------------

    mean_em_avg = (
        sum(
            x["EM_avg"]
            for x in evaluated
        )
        /
        total_q
    )

    mean_precision_em = (
        sum(
            x["Precision_EM"]
            for x in evaluated
        )
        /
        total_q
    )

    mean_recall_em = (
        sum(
            x["Recall_EM"]
            for x in evaluated
        )
        /
        total_q
    )

    mean_f1_em = (
        sum(
            x["F1_EM"]
            for x in evaluated
        )
        /
        total_q
    )

    mean_accuracy_em = (
        sum(
            x["Accuracy_EM"]
            for x in evaluated
        )
        /
        total_q
    )


    # --------------------------------------------------------
    # PEM
    # --------------------------------------------------------

    mean_pem_avg = (
        sum(
            x["PEM_avg"]
            for x in evaluated
        )
        /
        total_q
    )

    mean_precision_pem = (
        sum(
            x["Precision_PEM"]
            for x in evaluated
        )
        /
        total_q
    )

    mean_recall_pem = (
        sum(
            x["Recall_PEM"]
            for x in evaluated
        )
        /
        total_q
    )

    mean_f1_pem = (
        sum(
            x["F1_PEM"]
            for x in evaluated
        )
        /
        total_q
    )

    mean_accuracy_pem = (
        sum(
            x["Accuracy_PEM"]
            for x in evaluated
        )
        /
        total_q
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "=" * 60
    )

    print(
        "       SPIDER ANSWER EVALUATION"
    )

    print(
        "=" * 60
    )

    print(
        f"Gold test questions:      "
        f"{len(gold_data)}"
    )

    print(
        f"Predictions provided:     "
        f"{len(predictions)}"
    )

    print(
        f"Predictions evaluated:    "
        f"{total_q}"
    )

    print(
        f"Predictions unmatched:    "
        f"{unmatched}"
    )


    print(
        "\nIMPORTANT:"
    )

    print(
        "Only predictions present in "
        "--predictions_file are evaluated."
    )


    # --------------------------------------------------------
    # HIT
    # --------------------------------------------------------

    print(
        "\n--- Hit Rates ---"
    )

    print(
        f"Total EM HITs:             "
        f"{total_hit_em}"
    )

    print(
        f"Average EM HIT Rate:       "
        f"{mean_hit_em * 100:.2f}"
    )

    print(
        f"Total PEM HITs:            "
        f"{total_hit_pem}"
    )

    print(
        f"Average PEM HIT Rate:      "
        f"{mean_hit_pem * 100:.2f}"
    )


    # --------------------------------------------------------
    # EM
    # --------------------------------------------------------

    print(
        "\n--- Exact Match (EM) ---"
    )

    print(
        f"Mean EM Avg:               "
        f"{mean_em_avg * 100:.2f}"
    )

    print(
        f"Mean Precision:            "
        f"{mean_precision_em * 100:.2f}"
    )

    print(
        f"Mean Recall:               "
        f"{mean_recall_em * 100:.2f}"
    )

    print(
        f"Mean F1:                   "
        f"{mean_f1_em * 100:.2f}"
    )

    print(
        f"Mean Accuracy:             "
        f"{mean_accuracy_em * 100:.2f}"
    )


    # --------------------------------------------------------
    # PEM
    # --------------------------------------------------------

    print(
        "\n--- Partial Exact Match (PEM) ---"
    )

    print(
        f"Mean PEM Avg:              "
        f"{mean_pem_avg * 100:.2f}"
    )

    print(
        f"Mean Precision:            "
        f"{mean_precision_pem * 100:.2f}"
    )

    print(
        f"Mean Recall:               "
        f"{mean_recall_pem * 100:.2f}"
    )

    print(
        f"Mean F1:                   "
        f"{mean_f1_pem * 100:.2f}"
    )

    print(
        f"Mean Accuracy:             "
        f"{mean_accuracy_pem * 100:.2f}"
    )


    print(
        "\nDetailed results saved to:"
    )

    print(
        output_file
    )

    print(
        "=" * 60
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate only the predictions "
            "present in the prediction file."
        )
    )


    parser.add_argument(
        "--predictions_file",
        type=str,
        required=True,
        help=(
            "JSONL from 01_sql_to_answers.py "
            "(must contain predicted_answer)"
        )
    )


    parser.add_argument(
        "--gold_file",
        type=str,
        default=None,
        help=(
            "Optional gold JSON/JSONL. If omitted, the "
            "gold `answer` stored in each prediction row "
            "is used."
        )
    )


    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Detailed evaluation JSON"
    )


    args = parser.parse_args()


    run_evaluation(
        args.predictions_file,
        args.gold_file,
        args.output_file
    )