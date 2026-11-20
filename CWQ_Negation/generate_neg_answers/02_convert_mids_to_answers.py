import argparse
import json
import os
import time

import requests
from tqdm import tqdm


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Convert Freebase MIDs to English names"
)

parser.add_argument(
    "--input",
    required=True,
    help="Input JSON containing Freebase MIDs"
)

parser.add_argument(
    "--output",
    required=True,
    help="Output JSON with English names"
)

parser.add_argument(
    "--endpoint",
    default="http://localhost:3030/freebase/query",
    help="Fuseki SPARQL endpoint"
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=100,
    help="Number of MIDs per SPARQL query"
)

parser.add_argument(
    "--timeout",
    type=int,
    default=60,
    help="SPARQL timeout in seconds"
)

args = parser.parse_args()


# ============================================================
# CONFIGURATION
# ============================================================

print("\n" + "=" * 70)
print("FREEBASE MID -> ENGLISH NAME CONVERSION")
print("=" * 70)

print("Input      :", args.input)
print("Output     :", args.output)
print("Endpoint   :", args.endpoint)
print("Batch size :", args.batch_size)
print("Timeout    :", args.timeout)


# ============================================================
# LOAD JSON
# ============================================================

with open(
    args.input,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


if not isinstance(data, list):

    raise ValueError(
        "Input JSON must contain a list of examples."
    )


print(
    "\nTotal examples:",
    len(data)
)


# ============================================================
# EXTRACT MID
# ============================================================

def extract_mid(value):

    if not isinstance(value, str):
        return None

    value = value.strip()

    # Full Freebase URI
    prefix = "http://rdf.freebase.com/ns/"

    if value.startswith(prefix):

        return value[len(prefix):]


    # Already in m.xxxxx format
    if value.startswith("m."):

        return value


    # /m/xxxxx format
    if value.startswith("/m/"):

        return "m." + value[3:].replace(
            "/",
            "."
        )


    return None


# ============================================================
# COLLECT UNIQUE MIDS
# ============================================================

all_mids = set()

for example in data:

    answers = example.get(
        "negated_answers",
        []
    )

    if not isinstance(
        answers,
        list
    ):
        continue


    for answer in answers:

        mid = extract_mid(
            answer
        )

        if mid:

            all_mids.add(mid)


print(
    "Unique MIDs:",
    len(all_mids)
)


# ============================================================
# HTTP SESSION
# ============================================================

session = requests.Session()

session.headers.update({
    "Accept": "application/sparql-results+json"
})


# ============================================================
# QUERY ENGLISH NAMES
# ============================================================

def get_english_names(mids):

    if not mids:

        return {}


    values = []

    for mid in mids:

        values.append(
            f"ns:{mid}"
        )


    values_string = " ".join(
        values
    )


    # ========================================================
    # IMPORTANT:
    #
    # Only English or language-neutral names are requested.
    #
    # lang(?name) = "en"
    #
    # OR
    #
    # lang(?name) = ""
    #
    # ========================================================

    query = f"""
PREFIX ns: <http://rdf.freebase.com/ns/>

SELECT ?entity ?name
WHERE {{

    VALUES ?entity {{
        {values_string}
    }}

    ?entity ns:type.object.name ?name .

    FILTER(
        lang(?name) = "en"
        ||
        lang(?name) = ""
    )
}}
"""


    try:

        response = session.post(
            args.endpoint,

            data={
                "query": query
            },

            timeout=args.timeout
        )


        response.raise_for_status()

        result = response.json()


    except Exception as e:

        print(
            "\nSPARQL ERROR:"
        )

        print(
            str(e)
        )

        return {}


    # ========================================================
    # BUILD MID -> ENGLISH NAME
    # ========================================================

    mapping = {}


    bindings = (
        result
        .get(
            "results",
            {}
        )
        .get(
            "bindings",
            []
        )
    )


    for binding in bindings:

        if (
            "entity" not in binding
            or "name" not in binding
        ):

            continue


        entity_uri = (
            binding["entity"]
            .get("value")
        )


        name = (
            binding["name"]
            .get("value")
        )


        if not entity_uri or not name:

            continue


        mid = extract_mid(
            entity_uri
        )


        if not mid:

            continue


        # ----------------------------------------------------
        # Keep first English name
        # ----------------------------------------------------

        if mid not in mapping:

            mapping[mid] = name


    return mapping


# ============================================================
# RESOLVE MIDS IN BATCHES
# ============================================================

mids_list = list(
    all_mids
)

mid_to_name = {}


total_batches = (
    len(mids_list)
    + args.batch_size
    - 1
) // args.batch_size


print("\n" + "=" * 70)
print("RESOLVING ENGLISH NAMES")
print("=" * 70)

print(
    "Total batches:",
    total_batches
)


for start in tqdm(
    range(
        0,
        len(mids_list),
        args.batch_size
    ),
    desc="Resolving MIDs"
):

    batch = mids_list[
        start:start + args.batch_size
    ]


    mapping = get_english_names(
        batch
    )


    mid_to_name.update(
        mapping
    )


    time.sleep(0.1)


# ============================================================
# RESULTS
# ============================================================

unresolved = (
    all_mids
    - set(mid_to_name.keys())
)


print("\n" + "=" * 70)
print("RESOLUTION COMPLETE")
print("=" * 70)

print(
    "Total MIDs      :",
    len(all_mids)
)

print(
    "English resolved:",
    len(mid_to_name)
)

print(
    "Unresolved      :",
    len(unresolved)
)


if unresolved:

    print("\nFirst unresolved MIDs:")

    for mid in list(unresolved)[:20]:

        print(
            " ",
            mid
        )


# ============================================================
# REPLACE MIDS WITH ENGLISH NAMES
# ============================================================

converted = 0
unchanged = 0


for example in tqdm(
    data,
    desc="Converting answers"
):

    answers = example.get(
        "negated_answers",
        []
    )


    if not isinstance(
        answers,
        list
    ):

        continue


    new_answers = []


    for answer in answers:

        mid = extract_mid(
            answer
        )


        # ----------------------------------------------------
        # Not a MID
        # ----------------------------------------------------

        if not mid:

            new_answers.append(
                answer
            )

            unchanged += 1

            continue


        # ----------------------------------------------------
        # English name found
        # ----------------------------------------------------

        if mid in mid_to_name:

            new_answers.append(
                mid_to_name[mid]
            )

            converted += 1

            continue


        # ----------------------------------------------------
        # No English name found
        # ----------------------------------------------------

        new_answers.append(
            answer
        )

        unchanged += 1


    # Remove duplicate names while preserving order
    cleaned_answers = []

    for answer in new_answers:

        if answer not in cleaned_answers:

            cleaned_answers.append(
                answer
            )


    example[
        "negated_answers"
    ] = cleaned_answers


# ============================================================
# SAVE OUTPUT
# ============================================================

output_dir = os.path.dirname(
    os.path.abspath(
        args.output
    )
)

os.makedirs(
    output_dir,
    exist_ok=True
)


with open(
    args.output,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        data,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(
    "Examples processed :",
    len(data)
)

print(
    "MIDs found         :",
    len(all_mids)
)

print(
    "MIDs resolved      :",
    len(mid_to_name)
)

print(
    "Answers converted  :",
    converted
)

print(
    "Answers unchanged  :",
    unchanged
)

print(
    "\nOutput:"
)

print(
    args.output
)

print("=" * 70)