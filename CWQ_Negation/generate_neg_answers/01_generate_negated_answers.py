import argparse
import io
import json
import os
import time

import pandas as pd
import requests
from tqdm import tqdm


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Generate negated answers in small batches"
)

parser.add_argument(
    "--input",
    required=True,
    help="Input CSV file"
)

parser.add_argument(
    "--output",
    required=True,
    help="Successful output JSON"
)

parser.add_argument(
    "--errors",
    required=True,
    help="TIMEOUT/ERROR output JSON"
)

parser.add_argument(
    "--start",
    type=int,
    default=0,
    help="Starting row index (default: 0)"
)

parser.add_argument(
    "--limit",
    type=int,
    default=200,
    help="Number of rows to process (default: 200)"
)

parser.add_argument(
    "--endpoint",
    default="http://localhost:3030/freebase/query",
    help="Fuseki SPARQL endpoint"
)

parser.add_argument(
    "--timeout",
    type=int,
    default=90,
    help="Timeout per SPARQL query in seconds"
)

parser.add_argument(
    "--retries",
    type=int,
    default=0,
    help="Number of retries after failure"
)

parser.add_argument(
    "--save_every",
    type=int,
    default=25,
    help="Save progress every N rows"
)

args = parser.parse_args()


# ============================================================
# CONFIGURATION
# ============================================================

print("\n" + "=" * 70)
print("CONFIGURATION")
print("=" * 70)

print(f"Input       : {args.input}")
print(f"Output      : {args.output}")
print(f"Errors      : {args.errors}")
print(f"Start row   : {args.start}")
print(f"Batch size  : {args.limit}")
print(f"Timeout     : {args.timeout}s")
print(f"Retries     : {args.retries}")
print(f"Save every  : {args.save_every}")
print(
    "SPARQL mode : ORIGINAL QUERY — NO MODIFICATION"
)


# ============================================================
# VALIDATE START / LIMIT
# ============================================================

if args.start < 0:
    raise ValueError(
        "--start cannot be negative."
    )

if args.limit <= 0:
    raise ValueError(
        "--limit must be greater than 0."
    )


# ============================================================
# CREATE DIRECTORIES
# ============================================================

output_dir = os.path.dirname(
    os.path.abspath(args.output)
)

error_dir = os.path.dirname(
    os.path.abspath(args.errors)
)

os.makedirs(
    output_dir,
    exist_ok=True
)

os.makedirs(
    error_dir,
    exist_ok=True
)


# ============================================================
# LOAD CSV
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATASET")
print("=" * 70)

def read_csv_any_encoding(path):
    """Read a CSV that may be UTF-8, Windows cp1252/latin-1, or a mix of both.

    Each line is decoded as UTF-8 when it is valid UTF-8, otherwise as cp1252
    (falling back to latin-1 for the few bytes cp1252 leaves undefined).
    This avoids both the UnicodeDecodeError and the mojibake you get when a
    whole mixed file is decoded with a single encoding.
    """
    with open(path, "rb") as f:
        raw = f.read()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]                                   # strip UTF-8 BOM
    lines, n_fallback = [], 0
    for line in raw.split(b"\n"):
        try:
            lines.append(line.decode("utf-8"))
        except UnicodeDecodeError:
            n_fallback += 1
            try:
                lines.append(line.decode("cp1252"))
            except UnicodeDecodeError:
                lines.append(line.decode("latin-1"))
    if n_fallback:
        print(f"Note: {n_fallback} line(s) were not UTF-8 and were decoded as cp1252/latin-1")
    return pd.read_csv(io.StringIO("\n".join(lines)))


df = read_csv_any_encoding(args.input)

total_rows = len(df)

print(
    f"Total rows : {total_rows}"
)


# ============================================================
# CHECK BATCH RANGE
# ============================================================

if args.start >= total_rows:

    print(
        f"\nStart row {args.start} is beyond "
        f"the dataset ({total_rows} rows)."
    )

    print("Nothing to process.")
    exit(0)


batch_end = min(
    args.start + args.limit,
    total_rows
)

batch_df = df.iloc[
    args.start:batch_end
]

print("\n" + "=" * 70)
print("CURRENT BATCH")
print("=" * 70)

print(
    f"Start row : {args.start}"
)

print(
    f"End row   : {batch_end - 1}"
)

print(
    f"Batch size: {len(batch_df)}"
)


# ============================================================
# FIND SPARQL COLUMN
# ============================================================

if "negated_sparql" in df.columns:

    sparql_column = "negated_sparql"

elif "negated_filter" in df.columns:

    sparql_column = "negated_filter"

else:

    raise ValueError(
        "CSV does not contain "
        "'negated_sparql' or 'negated_filter'."
    )

print(
    f"\nSPARQL column: {sparql_column}"
)


# ============================================================
# IMPORTANT
# ============================================================
#
# The query will be sent to Fuseki EXACTLY as stored in CSV.
#
# NO:
#
# FILTER NOT EXISTS -> MINUS
# Query rewriting
# Query simplification
# Query optimization
# Query reordering
#
# ============================================================


# ============================================================
# HTTP SESSION
# ============================================================

session = requests.Session()

session.headers.update({
    "Accept": "application/sparql-results+json"
})


# ============================================================
# NORMALIZE VALUE FOR RECORD KEY
# ============================================================

def normalize_value(value):

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return str(value).strip()


# ============================================================
# CREATE UNIQUE RECORD KEY
# ============================================================

def get_record_key(row):

    """
    Prefer ID when available.

    If there is no ID, use:
        original_question + negated_question

    This prevents duplicate records when rerunning a batch.
    """

    if "id" in df.columns:

        row_id = row.get("id")

        if (
            row_id is not None
            and not pd.isna(row_id)
        ):

            return (
                "id::"
                + normalize_value(row_id)
            )

    original_question = normalize_value(
        row.get(
            "original_question",
            row.get(
                "question",
                ""
            )
        )
    )

    negated_question = normalize_value(
        row.get(
            "negated_question",
            ""
        )
    )

    return (
        "question::"
        + original_question
        + "||"
        + negated_question
    )


def get_record_key_from_record(record):

    if record.get("id") is not None:

        return (
            "id::"
            + normalize_value(
                record.get("id")
            )
        )

    original_question = normalize_value(
        record.get(
            "original_question",
            record.get(
                "question",
                ""
            )
        )
    )

    negated_question = normalize_value(
        record.get(
            "negated_question",
            ""
        )
    )

    return (
        "question::"
        + original_question
        + "||"
        + negated_question
    )


# ============================================================
# CONVERT PANDAS ROW TO JSON-SAFE DICT
# ============================================================

def row_to_dict(row):

    record = {}

    for key, value in row.items():

        if pd.isna(value):

            record[key] = None

        else:

            # Convert numpy values to normal Python values
            try:
                record[key] = value.item()
            except Exception:
                record[key] = value

    return record


# ============================================================
# LOAD JSON LIST
# ============================================================

def load_json_list(filepath):

    if not os.path.exists(filepath):

        return []

    try:

        with open(
            filepath,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        if isinstance(data, list):

            return data

        print(
            f"WARNING: {filepath} "
            f"does not contain a JSON list."
        )

        return []

    except Exception as e:

        print(
            f"WARNING: Could not read {filepath}"
        )

        print(
            f"Reason: {e}"
        )

        return []


# ============================================================
# LOAD EXISTING SUCCESSFUL OUTPUT
# ============================================================

successful_records = load_json_list(
    args.output
)

print("\n" + "=" * 70)
print("EXISTING SUCCESSFUL OUTPUT")
print("=" * 70)

print(
    f"Existing successful records: "
    f"{len(successful_records)}"
)


# ============================================================
# LOAD EXISTING ERRORS
# ============================================================

error_records = load_json_list(
    args.errors
)

print("\n" + "=" * 70)
print("EXISTING ERROR OUTPUT")
print("=" * 70)

print(
    f"Existing error records: "
    f"{len(error_records)}"
)


# ============================================================
# BUILD SUCCESS KEY SET
# ============================================================

successful_keys = set()

for record in successful_records:

    successful_keys.add(
        get_record_key_from_record(
            record
        )
    )


# ============================================================
# BUILD ERROR DICTIONARY
# ============================================================

def get_record_key_from_record(record):

    if record.get("id") is not None:

        return (
            "id::"
            + normalize_value(
                record.get("id")
            )
        )

    original_question = normalize_value(
        record.get(
            "original_question",
            record.get(
                "question",
                ""
            )
        )
    )

    negated_question = normalize_value(
        record.get(
            "negated_question",
            ""
        )
    )

    return (
        "question::"
        + original_question
        + "||"
        + negated_question
    )


# Rebuild because function is defined above now
successful_keys = set()

for record in successful_records:

    successful_keys.add(
        get_record_key_from_record(
            record
        )
    )


# ============================================================
# ERROR DICTIONARY
# ============================================================
#
# Using a dictionary means the same failed example will NOT
# be duplicated if the batch is rerun.
#
# ============================================================

error_dict = {}

for record in error_records:

    key = get_record_key_from_record(
        record
    )

    error_dict[key] = record


# ============================================================
# EXECUTE SPARQL
# ============================================================

def execute_sparql(query):

    last_error = None
    last_error_type = "ERROR"

    for attempt in range(
        args.retries + 1
    ):

        try:

            response = session.post(

                args.endpoint,

                # IMPORTANT:
                # ORIGINAL QUERY IS SENT AS-IS
                data={
                    "query": query
                },

                timeout=args.timeout
            )

            # ------------------------------------------------
            # HTTP ERROR
            # ------------------------------------------------

            if response.status_code >= 400:

                error_message = (
                    f"HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )

                last_error = error_message
                last_error_type = "HTTP_ERROR"

                # Retry server errors
                if (
                    response.status_code >= 500
                    and attempt < args.retries
                ):

                    wait_time = (
                        5 * (attempt + 1)
                    )

                    print(
                        f"\nServer error. "
                        f"Retrying in "
                        f"{wait_time}s..."
                    )

                    time.sleep(
                        wait_time
                    )

                    continue

                return {
                    "success": False,
                    "result": None,
                    "error_type": "HTTP_ERROR",
                    "error": error_message
                }


            # ------------------------------------------------
            # PARSE JSON RESPONSE
            # ------------------------------------------------

            result = response.json()

            return {
                "success": True,
                "result": result,
                "error_type": None,
                "error": None
            }


        # ----------------------------------------------------
        # TIMEOUT
        # ----------------------------------------------------

        except requests.exceptions.Timeout as e:

            last_error = str(e)

            last_error_type = "TIMEOUT"


        # ----------------------------------------------------
        # CONNECTION ERROR
        # ----------------------------------------------------

        except requests.exceptions.ConnectionError as e:

            last_error = str(e)

            last_error_type = "CONNECTION_ERROR"


        # ----------------------------------------------------
        # REQUEST ERROR
        # ----------------------------------------------------

        except requests.exceptions.RequestException as e:

            last_error = str(e)

            last_error_type = "REQUEST_ERROR"


        # ----------------------------------------------------
        # OTHER ERROR
        # ----------------------------------------------------

        except Exception as e:

            last_error = str(e)

            last_error_type = "ERROR"


        # ----------------------------------------------------
        # RETRY
        # ----------------------------------------------------

        if attempt < args.retries:

            wait_time = (
                5 * (attempt + 1)
            )

            print(
                f"\n{last_error_type}. "
                f"Retrying in "
                f"{wait_time}s..."
            )

            time.sleep(
                wait_time
            )


    return {
        "success": False,
        "result": None,
        "error_type": last_error_type,
        "error": last_error
    }


# ============================================================
# EXTRACT ANSWERS
# ============================================================

def extract_answers(result):

    if not result:

        return []


    results_section = result.get(
        "results",
        {}
    )

    bindings = results_section.get(
        "bindings",
        []
    )


    answers = []


    for binding in bindings:

        if not binding:
            continue


        # ----------------------------------------------------
        # Prefer ?x
        # ----------------------------------------------------

        if "x" in binding:

            value = (
                binding["x"]
                .get("value")
            )


        # ----------------------------------------------------
        # Otherwise take first variable
        # ----------------------------------------------------

        else:

            first_variable = next(
                iter(binding)
            )

            value = (
                binding[first_variable]
                .get("value")
            )


        if value is not None:

            if value not in answers:

                answers.append(value)


    return answers


# ============================================================
# SAVE PROGRESS
# ============================================================

def save_progress():

    # --------------------------------------------------------
    # SUCCESSFUL RECORDS
    # --------------------------------------------------------

    with open(
        args.output,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            successful_records,
            f,
            ensure_ascii=False,
            indent=2
        )


    # --------------------------------------------------------
    # ERROR RECORDS
    # --------------------------------------------------------

    current_errors = list(
        error_dict.values()
    )

    with open(
        args.errors,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            current_errors,
            f,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# STATISTICS
# ============================================================

batch_success = 0
batch_timeout = 0
batch_errors = 0

batch_zero_answers = 0
batch_over_10_answers = 0

batch_skipped = 0


# ============================================================
# PROCESS BATCH
# ============================================================

print("\n" + "=" * 70)
print("GENERATING NEGATED ANSWERS")
print("=" * 70)

print(
    "FILTER NOT EXISTS will remain unchanged."
)

print(
    f"Processing rows "
    f"{args.start} to {batch_end - 1}"
)

print()


for index, row in tqdm(
    batch_df.iterrows(),
    total=len(batch_df),
    desc="Executing SPARQL"
):

    # ========================================================
    # RECORD KEY
    # ========================================================

    record_key = get_record_key(
        row
    )


    # ========================================================
    # ALREADY SUCCESSFUL
    # ========================================================

    if record_key in successful_keys:

        batch_skipped += 1

        continue


    # ========================================================
    # GET ORIGINAL SPARQL
    # ========================================================

    query = row.get(
        sparql_column
    )


    # ========================================================
    # EMPTY SPARQL
    # ========================================================

    if (
        query is None
        or pd.isna(query)
        or not str(query).strip()
    ):

        record = row_to_dict(
            row
        )

        record[
            "negated_answers"
        ] = []

        record[
            "query_status"
        ] = "ERROR"

        record[
            "error_type"
        ] = "EMPTY_SPARQL"

        record[
            "error_message"
        ] = "SPARQL query is empty."


        error_dict[
            record_key
        ] = record


        batch_errors += 1

        continue


    # ========================================================
    # ORIGINAL QUERY
    # ========================================================

    query = str(
        query
    ).strip()


    # ========================================================
    # EXECUTE ORIGINAL QUERY
    # ========================================================

    result = execute_sparql(
        query
    )


    # ========================================================
    # QUERY FAILED
    # ========================================================

    if not result["success"]:

        record = row_to_dict(
            row
        )

        record[
            "negated_answers"
        ] = []

        record[
            "query_status"
        ] = result[
            "error_type"
        ]

        record[
            "error_type"
        ] = result[
            "error_type"
        ]

        record[
            "error_message"
        ] = result[
            "error"
        ]


        # ----------------------------------------------------
        # Update existing error rather than duplicate it
        # ----------------------------------------------------

        error_dict[
            record_key
        ] = record


        if (
            result["error_type"]
            == "TIMEOUT"
        ):

            batch_timeout += 1

        else:

            batch_errors += 1


        # Save immediately after failure
        save_progress()

        continue


    # ========================================================
    # EXTRACT ANSWERS
    # ========================================================

    answers = extract_answers(
        result["result"]
    )


    answer_count = len(
        answers
    )


    # ========================================================
    # BUILD SUCCESS RECORD
    # ========================================================

    record = row_to_dict(
        row
    )

    record[
        "negated_answers"
    ] = answers

    record[
        "query_status"
    ] = "SUCCESS"


    # ========================================================
    # ZERO ANSWERS
    # ========================================================

    if answer_count == 0:

        batch_zero_answers += 1

        # Do not add to successful JSON.
        #
        # Remove it from previous error JSON if it
        # previously failed.
        #
        error_dict.pop(
            record_key,
            None
        )

        continue


    # ========================================================
    # MORE THAN 10 ANSWERS
    # ========================================================

    if answer_count > 10:

        batch_over_10_answers += 1

        # Do not add to successful JSON.

        error_dict.pop(
            record_key,
            None
        )

        continue


    # ========================================================
    # 1-10 ANSWERS
    # ========================================================

    successful_records.append(
        record
    )

    successful_keys.add(
        record_key
    )

    # --------------------------------------------------------
    # If it previously existed as an error and now succeeds,
    # remove it from the error file.
    # --------------------------------------------------------

    error_dict.pop(
        record_key,
        None
    )


    batch_success += 1


    # ========================================================
    # CHECKPOINT
    # ========================================================

    rows_done = (
        index - args.start + 1
    )

    if (
        rows_done
        % args.save_every
        == 0
    ):

        save_progress()

        print(
            f"\nCheckpoint saved "
            f"after {rows_done} batch rows."
        )

        print(
            f"Successful total : "
            f"{len(successful_records)}"
        )

        print(
            f"Error total      : "
            f"{len(error_dict)}"
        )


# ============================================================
# FINAL SAVE
# ============================================================

save_progress()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("BATCH COMPLETED")
print("=" * 70)

print(
    f"Dataset total rows       : "
    f"{total_rows}"
)

print(
    f"Batch start              : "
    f"{args.start}"
)

print(
    f"Batch end                : "
    f"{batch_end - 1}"
)

print(
    f"Rows in this batch       : "
    f"{len(batch_df)}"
)

print("\n--- THIS BATCH ---")

print(
    f"New successful (1-10)   : "
    f"{batch_success}"
)

print(
    f"Skipped already success : "
    f"{batch_skipped}"
)

print(
    f"TIMEOUT                 : "
    f"{batch_timeout}"
)

print(
    f"Other errors            : "
    f"{batch_errors}"
)

print(
    f"0 answers discarded     : "
    f"{batch_zero_answers}"
)

print(
    f">10 answers discarded   : "
    f"{batch_over_10_answers}"
)

print("\n--- TOTAL FILES ---")

print(
    f"Total successful JSON   : "
    f"{len(successful_records)}"
)

print(
    f"Total error JSON        : "
    f"{len(error_dict)}"
)

print("\n" + "=" * 70)

print(
    "Successful output:"
)

print(
    args.output
)

print(
    "\nError output:"
)

print(
    args.errors
)

print("=" * 70)
print("DONE")
print("=" * 70)