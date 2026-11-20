import os
import json
import sqlite3
import argparse
import re
import time
from collections import Counter


SQL_KEY_CANDIDATES = [
    "predicted_sql",
    "pred_sql",
    "prediction",
    "generated_sql",
    "sql_pred",
    "output",
]


# ============================================================
# CLEAN SQL
# ============================================================

def clean_sql(sql):

    """Strip markdown code fences and surrounding whitespace."""

    sql = sql.strip()

    match = re.search(
        r"```(?:sql|sqlite)?\s*(.*?)```",
        sql,
        flags=re.DOTALL | re.IGNORECASE
    )

    if match:
        sql = match.group(1).strip()

    return sql


# ============================================================
# STRINGIFY CELLS
# ============================================================

def stringify_cell(value):

    """
    Match the format of the gold `answer` field in the
    prediction files, where every cell is str(value),
    e.g. 3.0 -> "3.0", 11198167.237771427 -> "11198167.237771427".
    SQL NULL stays as JSON null.
    """

    if value is None:
        return None

    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")

    return str(value)


# ============================================================
# MULTISET COMPARISON (for the sanity check only)
# ============================================================

def same_rows(a, b):

    if a is None or b is None:
        return False

    return (
        Counter(tuple(r) for r in a)
        == Counter(tuple(r) for r in b)
    )


# ============================================================
# JSON FALLBACK (BLOB columns return bytes)
# ============================================================

def json_default(obj):

    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="replace")

    return str(obj)


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(path):

    data = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            data.append(json.loads(line))

    return data


# ============================================================
# EXECUTE SQL
# ============================================================

def execute_sql(db_path, sql, timeout=30):

    conn = None

    try:

        # Read-only: a predicted DROP/DELETE/UPDATE must never
        # modify the gold Spider database.
        conn = sqlite3.connect(
            f"file:{db_path}?mode=ro",
            uri=True
        )

        # Abort queries that run longer than `timeout` seconds
        # (e.g. accidental cartesian joins).
        deadline = time.monotonic() + timeout

        conn.set_progress_handler(
            lambda: 1 if time.monotonic() > deadline else 0,
            10000
        )

        cursor = conn.cursor()

        cursor.execute(sql)

        rows = cursor.fetchall()

        rows = [
            [stringify_cell(v) for v in row]
            for row in rows
        ]

        return {
            "success": True,
            "answer": rows,
            "error": None
        }

    except Exception as e:

        error = str(e)

        if error == "interrupted":
            error = f"Timeout after {timeout}s"

        return {
            "success": False,
            "answer": None,
            "error": error
        }

    finally:

        if conn is not None:
            conn.close()


# ============================================================
# PROGRESS BAR
# ============================================================

def print_progress(current, total):

    percent = (
        current / total
    ) * 100

    bar_length = 40

    filled = int(
        bar_length *
        current /
        total
    )

    bar = (
        "█" * filled
        +
        "░" * (bar_length - filled)
    )

    print(
        f"\rExecution: "
        f"{percent:6.2f}% |"
        f"{bar}| "
        f"{current}/{total}",
        end="",
        flush=True
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Execute predicted Spider SQL "
            "and store ALL queries in one JSONL file."
        )
    )


    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Predictions JSONL file"
    )


    # --------------------------------------------------------
    # DATABASE DIRECTORY
    # --------------------------------------------------------

    parser.add_argument(
        "--database_dir",
        type=str,
        required=True,
        help="Spider database directory"
    )


    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help=(
            "JSONL file containing ALL predictions "
            "(successful + failed)"
        )
    )


    parser.add_argument(
        "--sql_key",
        type=str,
        default=None,
        help=(
            "Field holding the predicted SQL. "
            "Auto-detected if omitted."
        )
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=30,
        help="Per-query timeout in seconds"
    )


    args = parser.parse_args()


    # ========================================================
    # LOAD PREDICTIONS
    # ========================================================

    print(
        "Loading predictions..."
    )

    predictions = load_jsonl(
        args.input_file
    )

    total = len(
        predictions
    )

    print(
        f"Total predictions: {total}"
    )

    if total == 0:
        raise ValueError(
            f"No predictions found in {args.input_file}"
        )


    # ========================================================
    # DETECT SQL FIELD
    # ========================================================

    sql_key = args.sql_key

    if sql_key is None:

        sql_key = next(
            (
                k for k in SQL_KEY_CANDIDATES
                if k in predictions[0]
            ),
            None
        )

        if sql_key is None:
            raise KeyError(
                "Could not find a predicted-SQL field. "
                f"Keys in first record: "
                f"{list(predictions[0].keys())}. "
                "Pass --sql_key explicitly."
            )

    missing_key = sum(
        1 for p in predictions if sql_key not in p
    )

    print(
        f"SQL field          : {sql_key!r}"
        + (
            f"  (missing in {missing_key} records)"
            if missing_key else ""
        )
    )


    # ========================================================
    # CHECK DATABASE DIRECTORY
    # ========================================================

    if not os.path.isdir(
        args.database_dir
    ):

        raise FileNotFoundError(
            f"Database directory not found:\n"
            f"{args.database_dir}"
        )


    # ========================================================
    # REMOVE OLD OUTPUT FILE
    # ========================================================

    if os.path.abspath(args.output_file) == os.path.abspath(
        args.input_file
    ):

        raise ValueError(
            "--output_file must differ from --input_file"
        )

    if os.path.exists(
        args.output_file
    ):

        os.remove(
            args.output_file
        )


    # ========================================================
    # COUNTERS
    # ========================================================

    successful = 0
    failed = 0

    # Sanity checks against fields already in the input file
    status_checked = 0
    status_disagree = []
    match_checked = 0
    match_disagree = []


    # ========================================================
    # PROCESS PREDICTIONS
    # ========================================================

    with open(
        args.output_file,
        "a",
        encoding="utf-8"
    ) as output_file:

        for idx, prediction in enumerate(
            predictions
        ):

            db_id = prediction.get(
                "db_id"
            )

            predicted_sql = clean_sql(
                prediction.get(sql_key) or ""
            )


            # ------------------------------------------------
            # DATABASE PATH
            # ------------------------------------------------

            db_path = os.path.join(
                args.database_dir,
                str(db_id),
                f"{db_id}.sqlite"
            )


            # ------------------------------------------------
            # CHECK DATABASE
            # ------------------------------------------------

            if not db_id:

                result = {
                    "success": False,
                    "answer": None,
                    "error": "Missing db_id"
                }

            elif not os.path.exists(
                db_path
            ):

                result = {
                    "success": False,
                    "answer": None,
                    "error":
                        f"Database not found: {db_path}"
                }


            # ------------------------------------------------
            # CHECK EMPTY SQL
            # ------------------------------------------------

            elif not predicted_sql.strip():

                result = {
                    "success": False,
                    "answer": None,
                    "error": "Empty predicted SQL"
                }


            # ------------------------------------------------
            # EXECUTE SQL
            # ------------------------------------------------

            else:

                result = execute_sql(
                    db_path,
                    predicted_sql,
                    timeout=args.timeout
                )


            # =================================================
            # CREATE OUTPUT RECORD
            # =================================================

            output = dict(
                prediction
            )

            output[
                "executed_sql"
            ] = predicted_sql


            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            if result["success"]:

                successful += 1

                output[
                    "predicted_answer"
                ] = result["answer"]

                output[
                    "execution_success"
                ] = True

                output[
                    "pred_execution_error"
                ] = None


            # ------------------------------------------------
            # FAILURE
            # ------------------------------------------------

            else:

                failed += 1

                output[
                    "predicted_answer"
                ] = None

                output[
                    "execution_success"
                ] = False

                output[
                    "pred_execution_error"
                ] = result["error"]

                output[
                    "database_path"
                ] = db_path


            # =================================================
            # SANITY CHECKS AGAINST UPSTREAM FIELDS
            # =================================================

            row_id = prediction.get("id", idx)

            if "execution_error" in prediction:

                status_checked += 1

                upstream_ok = prediction["execution_error"] is None

                if upstream_ok != result["success"]:
                    status_disagree.append(row_id)

            if (
                prediction.get("execution_match") is True
                and "answer" in prediction
            ):

                match_checked += 1

                if not same_rows(
                    output["predicted_answer"],
                    prediction["answer"]
                ):
                    match_disagree.append(row_id)


            # =================================================
            # WRITE EVERY EXAMPLE
            # =================================================

            output_file.write(
                json.dumps(
                    output,
                    ensure_ascii=False,
                    default=json_default
                )
                + "\n"
            )

            # ------------------------------------------------
            # FORCE WRITE TO DISK
            # ------------------------------------------------

            output_file.flush()

            os.fsync(
                output_file.fileno()
            )


            # =================================================
            # PROGRESS
            # =================================================

            print_progress(
                idx + 1,
                total
            )


    # ========================================================
    # FINISHED
    # ========================================================

    print("\n")

    print(
        "=" * 60
    )

    print(
        "SQL execution completed."
    )

    print(
        f"Total predictions  : {total}"
    )

    print(
        f"Successful queries : {successful}"
    )

    print(
        f"Failed queries     : {failed}"
    )

    if status_checked:

        print()

        print(
            "Sanity check vs. upstream fields:"
        )

        print(
            f"  Success/failure agrees with "
            f"execution_error : "
            f"{status_checked - len(status_disagree)}"
            f"/{status_checked}"
        )

        print(
            f"  Answer equals gold where "
            f"execution_match : "
            f"{match_checked - len(match_disagree)}"
            f"/{match_checked}"
        )

        if status_disagree or match_disagree:

            print(
                "  Disagreements usually mean the database "
                "files differ from the ones used upstream."
            )

            print(
                f"  Example ids: "
                f"{list(dict.fromkeys(status_disagree + match_disagree))[:5]}"
            )

    print()

    print(
        f"ALL results:"
        f"\n{args.output_file}"
    )

    print(
        "=" * 60
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()