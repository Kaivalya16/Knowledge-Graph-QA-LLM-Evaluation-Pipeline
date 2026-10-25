import os
import json
import sqlite3
import argparse


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

def execute_sql(db_path, sql):

    conn = None

    try:

        conn = sqlite3.connect(db_path)

        cursor = conn.cursor()

        cursor.execute(sql)

        rows = cursor.fetchall()

        rows = [
            list(row)
            for row in rows
        ]

        return {
            "success": True,
            "answer": rows,
            "error": None
        }

    except Exception as e:

        return {
            "success": False,
            "answer": None,
            "error": str(e)
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

            predicted_sql = prediction.get(
                "predicted_sql",
                ""
            )


            # ------------------------------------------------
            # DATABASE PATH
            # ------------------------------------------------

            db_path = os.path.join(
                args.database_dir,
                db_id,
                f"{db_id}.sqlite"
            )


            # ------------------------------------------------
            # CHECK DATABASE
            # ------------------------------------------------

            if not os.path.exists(
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
                    predicted_sql
                )


            # =================================================
            # CREATE OUTPUT RECORD
            # =================================================

            output = dict(
                prediction
            )


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
                    "execution_error"
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
                    "execution_error"
                ] = result["error"]

                output[
                    "database_path"
                ] = db_path


            # =================================================
            # WRITE EVERY EXAMPLE
            # =================================================

            output_file.write(
                json.dumps(
                    output,
                    ensure_ascii=False
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