"""
Stage 2 (v2): generate NEGATED QA candidates from the canonical positives.

For every canonical positive, each VALUE predicate of its SQL (a column
compared with a literal or a subquery) is negated once, and the question is
rewritten so that its "not" lands on that same predicate:

    positive  Show the names of singers from France.
              SELECT name FROM singer WHERE country = 'France'
    negated   Show the names of singers not from France.
              SELECT name FROM singer WHERE country != 'France'

Changes from v1:

  1. The question and the SQL are negated TOGETHER. v1 flipped one SQL
     operator and, independently, put "not" after the first is/are/has/do of
     the question. That produced "What is not the average age ..." (1,050
     rows dropped later as head-attached) and left imperative questions
     ("List ...", "Show ...") unchanged (1,149 questions lost). The rewrite
     rules come from Stage 6 and now run at generation time, anchored on the
     predicate's own literal in the question.
  2. JOIN keys (`T1.id = T2.id`) are never negated. v1 negated them first,
     which is why Stage 5 had to relocate the negation and why Stage 3's
     filters were checking SQL that was later rewritten.
  3. Comparisons (>, <, >=, <=) are negated (561 questions had no v1
     candidate because their only condition was a comparison).
  4. A two-part name ("student Linda Smith" = first_name AND last_name) is
     negated as one entity: NOT (first_name = 'Linda' AND last_name = 'Smith').
  5. Each generated question passes quality gates before it becomes a
     candidate: it contains exactly one more negation word than the positive,
     it is not head-attached, and it matches no known ungrammatical pattern.
  6. Negated answers longer than MAX_STORED_ROWS are truncated in the file
     (the full row count is kept), so the candidates file stays small;
     whether the answer equals the positive answer is computed here on the
     full result.
  7. A JSON report with per-question and per-predicate counts is written.
  8. Coordination convention: when the question joins conditions with
     and / or / but / nor, only the LAST condition is negated
     ("bigger than 1500 or not smaller than 500", "in London or not
     Glasgow", "registered courses and did not attend courses"). Candidates
     that negate an earlier conjunct are rejected as `not_last_conjunct`.

Requirements: tqdm, and optionally lemminflect (pip install lemminflect) for
verb forms such as "taught" -> "did not teach". Without lemminflect the
script still runs, with a few more predicates left without a question rule.

The output keeps every field v1's Stage 3 reads, so the old Stage 3 still
runs on it; the new Stage 3 uses the extra fields.
"""

import os
import re
import json
import time
import sqlite3
import collections

from tqdm import tqdm

try:  # optional: better verb forms ("taught" -> "did not teach")
    from lemminflect import getLemma, getAllLemmas
except ImportError:  # pragma: no cover
    getLemma = getAllLemmas = None


# ============================================================
# PATH CONFIGURATION
# ============================================================

INPUT_DATASETS = {
    "train_spider": "data/processed/canonical_positive/canonical_positive_train_spider.json",
    "train_others": "data/processed/canonical_positive/canonical_positive_train_others.json",
    "dev": "data/processed/canonical_positive/canonical_positive_dev.json",
}

DATABASE_DIR = "data/raw/database"
OUTPUT_DIR = "data/processed/negation_candidates"
REPORT_FILE = os.path.join(OUTPUT_DIR, "stage2_report.json")

QUERY_TIMEOUT_SECONDS = 60
MAX_STORED_ROWS = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SQL: FIND VALUE PREDICATES
# ============================================================

IDENT = r"[A-Za-z_][A-Za-z0-9_]*"

OPERATOR_PATTERNS = [
    (r"\bNOT\s+BETWEEN\b", "NOT BETWEEN"),
    (r"\bBETWEEN\b", "BETWEEN"),
    (r"\bNOT\s+IN\b", "NOT IN"),
    (r"\bIN\b", "IN"),
    (r"\bNOT\s+LIKE\b", "NOT LIKE"),
    (r"\bLIKE\b", "LIKE"),
    (r"!=", "!="),
    (r"<>", "!="),
    (r">=", ">="),
    (r"<=", "<="),
    (r"(?<![<>=!])=(?!=)", "="),
    (r"(?<![<>=!])>(?!=)", ">"),
    (r"(?<![<>=!])<(?![=>])", "<"),
]

# operator -> (negated operator, negation_type)
NEGATION = {
    "=": ("!=", "equal_to_not_equal"),
    "IN": ("NOT IN", "in_to_not_in"),
    "LIKE": ("NOT LIKE", "like_to_not_like"),
    "BETWEEN": ("NOT BETWEEN", "between_to_not_between"),
    ">": ("<=", "greater_to_not_greater"),
    "<": (">=", "less_to_not_less"),
    ">=": ("<", "at_least_to_not_at_least"),
    "<=": (">", "at_most_to_not_at_most"),
}


def mask_sql_literals(sql):
    """Blank out string literals, keeping character offsets identical."""
    return re.sub(
        r"'[^']*'|\"[^\"]*\"",
        lambda m: m.group()[0] + "x" * (len(m.group()) - 2) + m.group()[-1],
        sql,
    )


def left_operand(sql, start):
    before = sql[:start].rstrip()
    aggregate = re.search(r"(?i)\b(count|sum|avg|min|max)\s*\([^()]*\)\s*$", before)
    if aggregate:
        return aggregate.group().strip()
    column = re.search(r"(%s(?:\s*\.\s*%s)?)\s*$" % (IDENT, IDENT), before)
    return column.group(1).strip() if column else ""


def right_operand(sql, end, operator):
    after = sql[end:].lstrip()
    if after.startswith("("):
        depth = 0
        for index, char in enumerate(after):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return after[:index + 1]
        return after
    if operator in ("BETWEEN", "NOT BETWEEN"):
        pair = re.match(
            r"(?is)\s*('[^']*'|\"[^\"]*\"|[\w.]+)\s+AND\s+('[^']*'|\"[^\"]*\"|[\w.]+)",
            after,
        )
        return pair.group(0).strip() if pair else ""
    single = re.match(
        r"('[^']*'|\"[^\"]*\"|-?\d+(?:\.\d+)?|%s(?:\s*\.\s*%s)?)" % (IDENT, IDENT),
        after,
    )
    return single.group(1).strip() if single else ""


def is_literal(text):
    return bool(re.match(r"^('.*'|\".*\"|-?\d+(?:\.\d+)?)$", text, re.S))


def classify_operand(operator, right):
    """'value' = compared with data, 'join' = column-to-column link."""
    if right == "":
        return "other"
    if right.startswith("("):
        return "value"
    if operator in ("BETWEEN", "NOT BETWEEN"):
        return "value"
    if is_literal(right):
        return "value"
    return "join"


def find_operators(sql):
    masked = mask_sql_literals(sql)
    occupied = []
    found = []
    for pattern, canonical in OPERATOR_PATTERNS:
        for match in re.finditer(pattern, masked, re.I):
            start, end = match.span()
            if any(start < o_end and end > o_start for o_start, o_end in occupied):
                continue
            occupied.append((start, end))
            found.append({"start": start, "end": end, "op": canonical})
    found.sort(key=lambda o: o["start"])
    for operator in found:
        operator["lhs"] = left_operand(sql, operator["start"])
        operator["rhs"] = right_operand(sql, operator["end"], operator["op"])
        operator["kind"] = classify_operand(operator["op"], operator["rhs"])
    return found


def negatable_targets(operators):
    targets = []
    for operator in operators:
        if operator["kind"] != "value":
            continue
        if operator["op"] not in NEGATION:
            continue
        if operator["rhs"].strip("'\"").lower() == "null":
            continue
        if operator["rhs"].startswith("(") and not re.match(r"\(\s*SELECT\b", operator["rhs"], re.I):
            continue  # IN ('a', 'b') literal lists: no reliable question anchor
        targets.append(operator)
    return targets


def predicate_span(sql, operator):
    before = sql[:operator["start"]].rstrip()
    start = len(before) - len(operator["lhs"])
    after = sql[operator["end"]:]
    padding = len(after) - len(after.lstrip())
    end = operator["end"] + padding + len(operator["rhs"])
    return start, end


def negate_operator(sql, operator):
    replacement, _ = NEGATION[operator["op"]]
    return sql[:operator["start"]] + replacement + sql[operator["end"]:]


# ============================================================
# QUESTION: TOKENS, MASKING, LITERAL LOCATION
# ============================================================

WORD = re.compile(r"[A-Za-z0-9_]+")

QUESTION_QUOTE_RE = re.compile(
    r"\"[^\"]*\"|“[^”]*”|‘[^’]*’|(?<![\w'])'[^']+'(?![\w])"
)

NEGATION_WORDS = re.compile(
    r"\b(?:not|no|never|without|none|neither|nor|cannot)\b|n't\b", re.I
)

NUMBER_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten",
}


def count_negations(question):
    masked = QUESTION_QUOTE_RE.sub(" QVAL ", question)
    return len(NEGATION_WORDS.findall(masked))


def words_before(question, position):
    return [(m.group().lower(), m.start(), m.end()) for m in WORD.finditer(question[:position])]


def insert(question, position, text):
    return question[:position] + text + question[position:]


def clean(question):
    question = re.sub(r"\s+", " ", question).strip()
    question = re.sub(r"\s+([?.,;])", r"\1", question)
    return question


def literal_text(operator):
    text = operator["rhs"].strip("'\"")
    if operator["op"] == "LIKE":
        text = text.strip("%").strip()
    return text


def locate_literal(question, literal):
    """Character span of a literal inside the question (quotes included)."""
    if not literal:
        return None
    candidates = [literal]
    if literal in NUMBER_WORDS:
        candidates.append(NUMBER_WORDS[literal])
    for value in candidates:
        for pattern in (r"(?<![\w])%s(?![\w])" % re.escape(value), re.escape(value)):
            match = re.search(pattern, question, re.I)
            if match:
                start, end = match.span()
                # widen to surrounding quotes so "not" goes outside them
                if start > 0 and question[start - 1] in "'\"‘“" and end < len(question) \
                        and question[end] in "'\"’”":
                    start, end = start - 1, end + 1
                return start, end
    return None


# ============================================================
# QUESTION REWRITE RULES
#
# Every rule makes one local edit next to the predicate's anchor in the
# question and returns None when it cannot do so safely.
# ============================================================

COPULAS = {"is", "are", "was", "were"}
AUXILIARIES = {"is", "are", "was", "were", "has", "have", "had", "can", "could",
               "will", "would", "do", "does", "did"}
MODALS = {"can", "could", "will", "would", "may", "might", "shall", "should", "must"}
RELATIVES = {"that", "who", "which", "whose", "whom"}
FRONTABLE_PREPOSITIONS = {"in", "from", "at", "on", "by", "to", "during", "near",
                          "under", "over"}
SCOPE_BREAKERS = {"and", "or", "both", "either", "neither", "that", "who", "which",
                  "whose", "with", "of", "having"}
BAD_ANCHOR_NEIGHBOURS = {"than", "there", "most", "least", "more", "less", "all",
                         "both", "either", "neither", "same", "as"}
UNSUPPORTED_VERBS = {"took", "equals", "equal", "equalling", "equaling", "became",
                     "went", "came", "gave", "saw", "said", "told", "brought", "began"}
PARTICIPLES = {"born", "located", "directed", "made", "produced", "held", "based",
               "written", "released", "sung", "founded", "situated", "stationed",
               "employed", "hired", "assigned", "accessed", "shipped", "billed",
               "registered", "enrolled", "opened", "closed", "built", "issued",
               "performed", "played", "hosted", "owned", "run", "managed", "led",
               "taught", "given", "sold", "bought", "used", "sent", "paid", "listed",
               "described", "named", "called", "titled", "labelled", "labeled"}
PAST_PARTICIPLES = {"been", "had", "gone", "got", "made", "taken", "done", "won",
                    "seen", "given", "written", "held", "sold", "bought", "paid",
                    "sent", "kept", "run"}
IRREGULAR_PAST = {"built", "made", "won", "took", "gave", "held", "sold", "bought",
                  "taught", "brought", "wrote", "drove", "ran", "led", "spent", "sent",
                  "paid", "kept", "left", "met", "began", "chose", "flew", "grew"}
CONTRASTABLE_NOUNS = {
    "code", "codes", "id", "ids", "name", "names", "number", "numbers", "type",
    "types", "status", "title", "titles", "grade", "grades", "course", "courses",
    "room", "rooms", "district", "districts", "level", "levels", "format",
    "formats", "category", "categories", "gender", "sex", "color", "colour",
    "language", "languages", "country", "countries", "city", "cities", "state",
    "states", "region", "regions", "currency", "nationality", "position",
    "positions", "department", "departments", "year", "years", "month", "day",
    "rank", "ranking", "size", "brand", "model", "models", "make", "maker",
    "sport", "team", "league", "continent", "party", "class", "role", "roles",
    "service", "services", "product", "products", "company", "companies",
}
SINGULAR_IN_S = {"status", "address", "class", "business", "series", "gas"}
PHRASE_ENDERS = re.compile(
    r"^\s*($|[?.,;)]|\b(and|or|but|who|that|which|whose|with|in|on|at|for|from|by|"
    r"to|of|sorted|ordered|order|list|give|return|show)\b)", re.I
)


def do_support(verb):
    """"reports" -> "does not report", "belong" -> "do not belong"."""
    if verb.endswith("ies") and len(verb) > 4:
        return "does not " + verb[:-3] + "y"
    if verb.endswith(("ses", "xes", "zes", "ches", "shes")):
        return "does not " + verb[:-2]
    if verb.endswith("s") and not verb.endswith("ss"):
        return "does not " + verb[:-1]
    return "do not " + verb


FUNCTION_WORDS = {
    "the", "a", "an", "of", "in", "on", "at", "by", "to", "for", "from", "with",
    "and", "or", "but", "is", "are", "was", "were", "be", "been", "has", "have",
    "had", "do", "does", "did", "that", "who", "which", "whose", "whom", "what",
    "where", "when", "how", "all", "each", "every", "some", "any", "both",
    "either", "neither", "as", "than", "more", "less", "most", "least", "not",
    "no", "their", "its", "his", "her", "there", "it", "they", "many", "much",
    "same", "only", "exactly", "also", "called", "named", "one", "two",
    "three", "four", "five", "least", "people", "person",
}


def is_verb(word):
    if getAllLemmas is None:
        return word.endswith(("s", "ed")) or word in IRREGULAR_PAST
    return "VERB" in getAllLemmas(word)


def negate_finite_verb(word):
    """"hold" -> "do not hold", "holds" -> "does not hold",
    "taught" -> "did not teach". None when the form is unclear."""
    if getLemma is not None:
        lemma = getLemma(word, upos="VERB")[0]
        if lemma == word:
            return "do not " + word
        if word.endswith("s") and not word.endswith("ss") and (
                lemma + "s" == word or lemma + "es" == word or
                (lemma.endswith("y") and word.endswith("ies"))):
            return "does not " + lemma
        return "did not " + lemma
    if word in IRREGULAR_PAST or word.endswith("ed"):
        return None
    return do_support(word)


def some_to_any(question, start):
    """After a new negation, the first "some" of the same clause becomes
    "any": "who did not write some review" -> "who did not write any review"."""
    tail = question[start:]
    boundary = re.search(r"[,;?]|\b(?:and|or|but)\b", tail)
    limit = boundary.start() if boundary else len(tail)
    match = re.search(r"\bsome\b", tail[:limit], re.I)
    if not match:
        return question
    position = start + match.start()
    return question[:position] + "any" + question[position + 4:]


def rule_participle(question, span):
    """"the employee named Kelly" -> "the employee not named Kelly"."""
    preceding = words_before(question, span[0])
    while preceding and preceding[-1][0] in ("a", "an", "the", "as"):
        preceding = preceding[:-1]
    if not preceding:
        return None
    word, start, end = preceding[-1]
    previous = preceding[-2][0] if len(preceding) > 1 else ""
    before_previous = preceding[-3][0] if len(preceding) > 2 else ""
    if word not in PARTICIPLES:
        # a generic -ed word is a participle only after "the/a NOUN"
        # ("the product listed ..."), not after a bare subject
        # ("How many students received ..." is a finite verb)
        if not (word.endswith("ed") and len(word) > 4
                and before_previous in ("the", "a", "an", "all", "each", "every")):
            return None
    if previous in RELATIVES or previous in ("not", "never"):
        return None
    if previous in COPULAS:
        return insert(question, preceding[-2][2], " not")
    if previous in AUXILIARIES or previous in ("be", "been"):
        return None
    return insert(question, start, "not ")


CONJUNCTION_WORDS = {"and", "or", "but", "nor"}


def rule_coordination(question, span):
    """Last item of a coordination: "in London or Glasgow" ->
    "in London or not Glasgow", "'CD' or 'DVD'" -> "'CD' or not 'DVD'"."""
    preceding = words_before(question, span[0])
    for back in range(1, min(3, len(preceding)) + 1):
        word, _, end = preceding[-back]
        if word in CONJUNCTION_WORDS:
            skipped = [w for w, _, _ in preceding[-back + 1:]] if back > 1 else []
            if any(w not in FRONTABLE_PREPOSITIONS | {"a", "an", "the"} for w in skipped):
                return None
            return insert(question, end, " not")
        if word not in FRONTABLE_PREPOSITIONS | {"a", "an", "the"}:
            return None
    return None


def rule_copula(question, span):
    """"whose email is X" -> "whose email is not X"."""
    preceding = words_before(question, span[0])
    for back in (1, 2, 3):
        if len(preceding) < back:
            break
        word, _, end = preceding[-back]
        if word in COPULAS:
            skipped = [w for w, _, _ in preceding[-back + 1:]] if back > 1 else []
            if any(w not in ("a", "an", "the") for w in skipped):
                continue
            return insert(question, end, " not")
    return None


def rule_preposition(question, span):
    """"born in state 'Alabama'" -> "not born in state 'Alabama'"."""
    preceding = words_before(question, span[0])
    if not preceding:
        return None
    index = None
    for position in range(len(preceding) - 1, max(-1, len(preceding) - 5), -1):
        word = preceding[position][0]
        if word in SCOPE_BREAKERS:
            return None
        if word in FRONTABLE_PREPOSITIONS:
            index = position
            break
    if index is None:
        return None
    following = WORD.findall(question[span[1]:span[1] + 24])
    if following and following[0].lower() in ("and", "or"):
        return None
    previous = preceding[index - 1][0] if index > 0 else ""
    if previous in BAD_ANCHOR_NEIGHBOURS or previous in FRONTABLE_PREPOSITIONS:
        return None
    for position in range(index - 1, -1, -1):
        word = preceding[position][0]
        # a relative pronoun or a conjunction closes the phrase: "both ... in
        # 'USA' and companies with headquarters in 'China'" negates the last one
        if word in RELATIVES or word in CONJUNCTION_WORDS:
            break
        if word == "neither" or word in FRONTABLE_PREPOSITIONS:
            return None
    if previous in UNSUPPORTED_VERBS:
        return None
    before_previous = preceding[index - 2][0] if index >= 2 else ""
    if previous in PARTICIPLES or previous.endswith("ed") or previous.endswith("ing"):
        if before_previous in RELATIVES:
            return None
        if before_previous in ("be", "been"):
            for position in range(index - 3, max(-1, index - 5), -1):
                if preceding[position][0] in AUXILIARIES:
                    return insert(question, preceding[position][2], " not")
            return None
        return insert(question, preceding[index - 1][1], "not ")
    if previous in COPULAS:
        return insert(question, preceding[index - 1][2], " not")
    if before_previous in AUXILIARIES:
        return insert(question, preceding[index - 2][2], " not")
    if before_previous in RELATIVES and previous:
        start, end = preceding[index - 1][1], preceding[index - 1][2]
        negated = negate_finite_verb(previous)
        return question[:start] + negated + question[end:] if negated else None
    # "How many people live in X" -> "How many people do not live in X"
    if previous and previous not in FUNCTION_WORDS and getAllLemmas is not None \
            and set(getAllLemmas(previous)) == {"VERB"}:
        negated = negate_finite_verb(previous)
        if not negated:
            return None
        start, end = preceding[index - 1][1], preceding[index - 1][2]
        return question[:start] + negated + question[end:]
    return insert(question, preceding[index][1], "not ")


def rule_contrast(question, span):
    """"with type code 'Studio'" -> "with type code that is not 'Studio'"."""
    if not PHRASE_ENDERS.match(question[span[1]:]):
        return None
    following = WORD.findall(question[span[1]:span[1] + 24])
    if following and following[0].lower() in ("and", "or"):
        return None
    preceding = words_before(question, span[0])
    if not preceding:
        return None
    noun = preceding[-1][0]
    if noun not in CONTRASTABLE_NOUNS:
        # any other content word directly before the literal acts as its noun
        # ("with the description 'Book'"), unless it is a function word or verb
        if noun in FUNCTION_WORDS or noun.isdigit() or len(noun) < 3:
            return None
        if getAllLemmas is not None and set(getAllLemmas(noun)) == {"VERB"}:
            return None
    plural = noun.endswith("s") and noun not in SINGULAR_IN_S
    return insert(question, span[0], "that are not " if plural else "that is not ")


LIKE_CUES = {
    "containing": "not containing", "including": "not including",
    "starting": "not starting", "beginning": "not beginning",
    "ending": "not ending", "like": "not like", "with": "without",
}
LIKE_VERBS = {"contain", "contains", "include", "includes", "start", "starts",
              "begin", "begins", "end", "ends", "have", "has", "had"}


def rule_like(question, span):
    """"names containing 'Mary'" -> "names not containing 'Mary'"."""
    preceding = words_before(question, span[0])
    for back in range(1, min(5, len(preceding)) + 1):
        word, start, end = preceding[-back]
        if word in SCOPE_BREAKERS - {"with"} or word in ("and", "or"):
            return None
        if word in LIKE_CUES:
            # "starting with" -> "not starting with": the cue is the verb, not "with"
            if word == "with" and back < len(preceding) and \
                    (preceding[-back - 1][0] in LIKE_CUES or preceding[-back - 1][0] in LIKE_VERBS):
                continue
            return question[:start] + keep_case(question[start:end], LIKE_CUES[word]) + question[end:]
        if word in LIKE_VERBS:
            if word in ("have", "has", "had"):
                support = {"have": "do not have", "has": "does not have",
                           "had": "did not have"}[word]
            else:
                support = do_support(word)
            return question[:start] + keep_case(question[start:end], support) + question[end:]
    return None


def rule_between(question, span):
    """"price between 10 and 20" -> "price not between 10 and 20",
    "that have between 1 and 2 ..." -> "that do not have between 1 and 2 ..."."""
    preceding = words_before(question, span[0])
    for back in range(1, min(4, len(preceding)) + 1):
        word, start, _ = preceding[-back]
        if word != "between":
            continue
        if back == len(preceding):
            return None
        previous, p_start, p_end = preceding[-back - 1]
        if previous in COPULAS or previous in FUNCTION_WORDS - {"have", "has", "had"}:
            return insert(question, start, "not ")
        if previous in ("have", "has", "had"):
            support = {"have": "do not have", "has": "does not have", "had": "did not have"}[previous]
            return question[:p_start] + keep_case(question[p_start:p_end], support) + question[p_end:]
        if is_verb(previous) and not (getAllLemmas and "NOUN" in getAllLemmas(previous)):
            return None  # "projects that require between ..." -> no safe edit
        return insert(question, start, "not ")
    return None


COMPARATIVE_PHRASES = {
    ">": [r"\bmore than\b", r"\bgreater than\b", r"\bhigher than\b",
          r"\blarger than\b", r"\bolder than\b", r"\blonger than\b",
          r"\bbigger than\b", r"\bheavier than\b", r"\blater than\b",
          r"\babove\b", r"\bover\b", r"\bafter\b", r"\bexceeds?\b",
          r"\bexceeding\b"],
    ">=": [r"\bat least\b", r"\b(?:\d+|one|two|three|four|five|six)\s+or more\b",
           r"\bno less than\b", r"\bnot less than\b"],
    "<": [r"\bless than\b", r"\bfewer than\b", r"\byounger than\b",
          r"\bsmaller than\b", r"\blower than\b", r"\bshorter than\b",
          r"\blighter than\b", r"\bearlier than\b", r"\bbelow\b", r"\bunder\b",
          r"\bbefore\b"],
    "<=": [r"\bat most\b", r"\b(?:\d+|one|two|three|four|five|six)\s+or (?:fewer|less)\b",
           r"\bno more than\b", r"\bnot more than\b"],
}

LEXICAL_NEGATION = {
    "more than": "no more than", "greater than": "not greater than",
    "higher than": "not higher than", "larger than": "not larger than",
    "older than": "not older than", "longer than": "not longer than",
    "bigger than": "not bigger than", "heavier than": "not heavier than",
    "later than": "not later than", "above": "not above", "over": "not over",
    "after": "not after", "exceeding": "not exceeding",
    "less than": "not less than", "fewer than": "not fewer than",
    "younger than": "not younger than", "smaller than": "not smaller than",
    "lower than": "not lower than", "shorter than": "not shorter than",
    "lighter than": "not lighter than", "earlier than": "not earlier than",
    "below": "not below", "under": "not under", "before": "not before",
}

ALL_COMPARATIVES = re.compile("|".join(p for ps in COMPARATIVE_PHRASES.values() for p in ps), re.I)


def keep_case(original, replacement):
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def number_follows(question, end, literal):
    """True when the predicate's value appears right after the phrase
    (numbers, decimals, dates, number words; "15,000" matches 15000)."""
    window = re.sub(r"(?<=\d),(?=\d{3})", "", question[end:end + 45]).lower()
    window = " ".join(window.split()[:5])
    values = {literal.lower()}
    if literal in NUMBER_WORDS:
        values.add(NUMBER_WORDS[literal])
    if re.fullmatch(r"-?\d+\.0+", literal):
        values.add(literal.split(".")[0])
    if re.match(r"\d{4}-\d{2}-\d{2}", literal):
        values.add(literal[:10])
    return any(
        re.search(r"(?<![\w.])%s(?![\w])" % re.escape(value), window) for value in values
    )


def negate_nearest_verb(question, position):
    """Negate the auxiliary/verb just before a comparative phrase.

    "who have at least two accounts" -> "who do not have at least two accounts"
    """
    preceding = words_before(question, position)
    for back in range(1, min(4, len(preceding)) + 1):
        word, start, end = preceding[-back]
        if word in ("and", "or", "with", "of"):
            return None
        if word in COPULAS or word in MODALS:
            return insert(question, end, " not")
        if word in ("have", "has", "had"):
            support = {"have": "do not have", "has": "does not have", "had": "did not have"}[word]
            return question[:start] + keep_case(question[start:end], support) + question[end:]
        if back != 1 or word in FUNCTION_WORDS or not is_verb(word):
            return None
        previous = preceding[-2][0] if len(preceding) > 1 else ""
        # "who hold two or more cards"
        after_relative = previous in RELATIVES
        # "Which wineries produce at least four wines?"
        after_wh_subject = (
            preceding[0][0] in ("which", "what") and 2 < len(preceding) <= 4
            and all(w not in FUNCTION_WORDS for w, _, _ in preceding[1:-1])
        )
        if not (after_relative or after_wh_subject):
            return None
        negated = negate_finite_verb(word)
        if not negated:
            return None
        result = question[:start] + keep_case(question[start:end], negated) + question[end:]
        return some_to_any(result, start)
    return None


def rule_comparison(question, target):
    operator = target["op"]
    rhs = target["rhs"]
    literal = rhs.strip("'\"")
    matches = []
    for pattern in COMPARATIVE_PHRASES.get(operator, []):
        for match in re.finditer(pattern, question, re.I):
            matches.append(match)
    if not matches:
        return None
    if is_literal(rhs):
        matches = [m for m in matches if number_follows(question, m.end(), literal)]
    elif len(ALL_COMPARATIVES.findall(question)) != 1:
        return None  # subquery comparison with several comparatives: ambiguous
    if len(matches) != 1:
        return None
    match = matches[0]
    phrase = match.group().lower()
    if phrase.startswith(("no ", "not ")):
        return None  # already negative wording: would produce a double negation
    replacement = LEXICAL_NEGATION.get(phrase)
    if replacement:
        return question[:match.start()] + keep_case(match.group(), replacement) + question[match.end():]
    return negate_nearest_verb(question, match.start())


def rule_relative_clause(question):
    """"artists that have had exhibitions" -> "artists that have not had ...".

    Only a single, genuine relative clause qualifies: a wh-word that opens
    the question ("Which department has ...") is not a relative pronoun.
    """
    tokens = list(WORD.finditer(question))
    relatives = []
    for index, token in enumerate(tokens):
        if token.group().lower() not in RELATIVES:
            continue
        prefix = question[:token.start()].rstrip()
        if not prefix or prefix[-1] in ".?;:,":
            continue
        relatives.append(index)
    if len(relatives) != 1:
        return None
    index = relatives[0]
    if index + 1 >= len(tokens):
        return None
    # "who are older than 18 and have allergies": the subquery may belong to
    # either conjunct, so the attachment is ambiguous
    clause_start = tokens[index].end()
    clause = re.split(r"[,;?]", question[clause_start:])[0]
    conjunctions = list(re.finditer(r"\b(and|or|but)\b", clause, re.I))
    if conjunctions:
        return negate_last_conjunct(question, clause_start + conjunctions[-1].end(), clause_start + len(clause))
    auxiliary = tokens[index + 1]
    word = auxiliary.group().lower()
    following = tokens[index + 2].group().lower() if index + 2 < len(tokens) else ""
    if word in COPULAS or word in MODALS or word in ("do", "does", "did"):
        return some_to_any(insert(question, auxiliary.end(), " not"), auxiliary.start())
    if word in ("have", "has", "had"):
        if following in PAST_PARTICIPLES or following.endswith("ed"):
            return some_to_any(insert(question, auxiliary.end(), " not"), auxiliary.start())
        support = {"have": "do not have", "has": "does not have", "had": "did not have"}[word]
        result = question[:auxiliary.start()] + support + question[auxiliary.end():]
        return some_to_any(result, auxiliary.start())
    if word in FUNCTION_WORDS or not re.fullmatch(r"[a-z]+", word) or not is_verb(word):
        return None
    negated = negate_finite_verb(word)
    if not negated:
        return None
    result = question[:auxiliary.start()] + negated + question[auxiliary.end():]
    return some_to_any(result, auxiliary.start())


def negate_last_conjunct(question, start, end):
    """Membership in a conjoined relative clause belongs to the LAST conjunct:
    "who registered courses and attended courses" ->
    "who registered courses and did not attend courses".
    The last conjunct must be a plain verb phrase: no literal, number or
    comparison that another predicate could own."""
    conjunct = question[start:end]
    if re.search(r"[\"'“‘\d]", conjunct) or ALL_COMPARATIVES.search(conjunct):
        return None
    tokens = list(WORD.finditer(conjunct))
    if not tokens:
        return None
    verb = tokens[0]
    word = verb.group().lower()
    following = tokens[1].group().lower() if len(tokens) > 1 else ""
    v_start, v_end = start + verb.start(), start + verb.end()
    if word in COPULAS or word in MODALS or word in ("do", "does", "did"):
        return some_to_any(insert(question, v_end, " not"), v_start)
    if word in ("have", "has", "had"):
        if following in PAST_PARTICIPLES or following.endswith("ed"):
            return some_to_any(insert(question, v_end, " not"), v_start)
        support = {"have": "do not have", "has": "does not have", "had": "did not have"}[word]
        return some_to_any(question[:v_start] + support + question[v_end:], v_start)
    if word in FUNCTION_WORDS or not is_verb(word):
        return None
    negated = negate_finite_verb(word)
    if not negated:
        return None
    return some_to_any(question[:v_start] + negated + question[v_end:], v_start)


def rewrite_question(question, target, span=None, subquery_targets=1):
    """Return (negated_question, rule_name) or (None, failure_reason)."""
    operator = target["op"]

    if operator in (">", "<", ">=", "<="):
        result = rule_comparison(question, target)
        return (clean(result), "comparison") if result else (None, "no_rule_fired")

    if target["rhs"].startswith("("):
        if subquery_targets != 1:
            return None, "ambiguous_subquery"
        result = rule_relative_clause(question)
        return (clean(result), "relative_clause") if result else (None, "no_rule_fired")

    if span is None:
        literal = literal_text(target)
        if operator == "BETWEEN":
            literal = re.split(r"(?i)\s+AND\s+", target["rhs"])[0].strip("'\"")
        span = locate_literal(question, literal)
    if span is None:
        return None, "literal_not_in_question"
    # "format is 'CD' or 'DVD'": negating one disjunct reads as negating both
    after = question[span[1]:]
    if operator != "BETWEEN" and (
            re.match(r"^[\s,]*(or|nor)\b", after, re.I)
            or re.match(r"^[\s,]*and\s+[\"'“‘\d]", after, re.I)):
        return None, "coordinated_literal"

    if operator == "BETWEEN":
        rules = [(rule_between, "between")]
    elif operator == "LIKE":
        rules = [(rule_like, "like"), (rule_copula, "copula"),
                 (rule_preposition, "preposition")]
    else:
        rules = [(rule_copula, "copula"), (rule_participle, "participle"),
                 (rule_preposition, "preposition"), (rule_contrast, "contrast"),
                 (rule_coordination, "coordination")]

    for rule, name in rules:
        result = rule(question, span)
        if result:
            return clean(result), name
    return None, "no_rule_fired"


# ============================================================
# QUALITY GATES ON THE GENERATED QUESTION
# ============================================================

HEAD_DETERMINERS = {"the", "a", "an", "all", "its", "their"}
BARE_WH = re.compile(
    r"^(?:and\s+|so\s+|then\s+)?(?:in|on|at|for|of|by|with|from|to)?\s*"
    r"(?:(what|who|whom|which|where|when)|how\s+(old|long|tall|far|big|large|wide|"
    r"heavy|fast|deep|high))$"
)
WH_START = re.compile(r"^(?:in|on|at|for|of|by|with|from|to)?\s*(what|which|who|whom|whose|where|when|how)\b")
WH_NOUN_PHRASE = re.compile(r"^(?:in|on|at|for|of|by|with|from|to)?\s*(?:what|which)(?:\s+\w+){1,3}$")
SUPERLATIVES = {"most", "least", "largest", "smallest", "highest", "lowest", "longest",
                "shortest", "oldest", "youngest", "biggest", "greatest", "best",
                "worst", "first", "last", "top", "maximum", "minimum", "same", "only"}
STRANDED_PREPOSITION = re.compile(
    r"\b(in|from|to|for|on|with|at|by|about|of|into|under|over)\s*[?.]?\s*$", re.I
)

UNGRAMMATICAL = [
    re.compile(r"^\W*(which|what)\s+(do|does|did)\s+not\b", re.I),
    re.compile(r"\bnot\s+not\b", re.I),
    re.compile(r"\b(do|does|did)\s+not\s+(is|are|was|were|has|had|do|does|did|can|will)\b", re.I),
    re.compile(r"\b(is|are|was|were)\s+not\s+not\b", re.I),
    # "What is not Astrid Gruber's email?": the "not" negates the answer
    re.compile(r"^\W*(?i:what|who|which|where|when)\s+(?i:is|are|was|were)\s+not\s+"
               r"(?:(?i:the|a|an|all|its|their|his|her)\b|[A-Z])"),
]


def tokenize_question(question):
    return re.findall(r"[A-Za-z0-9_%]+|[,.;?]", question.lower())


def is_head_attached(positive_question, negated_question):
    """True when the inserted "not" negates the answer, not a condition."""
    positive = tokenize_question(positive_question)
    tokens = tokenize_question(negated_question)
    prefix = 0
    while prefix < len(positive) and prefix < len(tokens) and positive[prefix] == tokens[prefix]:
        prefix += 1
    index = None
    for position in range(max(0, prefix - 2), min(len(tokens), prefix + 3)):
        if tokens[position] == "not":
            index = position
            break
    if index is None:
        return False
    previous = tokens[index - 1] if index > 0 else ""
    following = tokens[index + 1] if index + 1 < len(tokens) else ""
    if previous not in COPULAS:
        return False
    if following == "there":
        return True
    if following not in HEAD_DETERMINERS:
        return False
    start = 0
    for position in range(index - 1):
        if tokens[position] in (",", ".", ";", "?"):
            start = position + 1
    subject = " ".join(tokens[start:index - 1]).strip()
    if BARE_WH.match(subject):
        return True
    if WH_START.match(subject) and STRANDED_PREPOSITION.search(negated_question):
        return True
    after_determiner = tokens[index + 2] if index + 2 < len(tokens) else ""
    if (following == "the" and after_determiner not in SUPERLATIVES
            and WH_NOUN_PHRASE.match(subject)
            and not set(subject.split()) & (RELATIVES - {"which"})):
        return True
    return False


LAST_CONJUNCT_EXEMPT = re.compile(
    r"^\s*(?:then\s+)?(?:order|sort|rank|list|show|give|return|display|group|"
    r"also|include|tell)\b", re.I
)


def edit_position(positive_question, negated_question):
    position = 0
    limit = min(len(positive_question), len(negated_question))
    while position < limit and positive_question[position].lower() == negated_question[position].lower():
        position += 1
    return position


def conjunction_after_edit(positive_question, negated_question):
    """
    Convention: when a question coordinates several conditions with
    and / or / but / nor, the negation goes on the LAST one. Returns True
    when a coordinating conjunction still follows the edit inside the same
    sentence. Ignored: conjunctions inside quoted values, the "and" of
    "between X and Y", and conjunctions that only add an instruction
    ("... and order them by name", "... and return their ids").
    """
    position = edit_position(positive_question, negated_question)
    text = QUESTION_QUOTE_RE.sub(lambda m: "Q" * len(m.group()), negated_question)
    text = re.sub(r"\bbetween\s+\S+\s+and\b", lambda m: "B" * len(m.group()), text, flags=re.I)
    boundary = re.search(r"[?!;]|\.(?=\s|$)", text[position:])
    sentence_end = position + boundary.start() if boundary else len(text)
    for match in re.finditer(r"\b(and|or|but|nor)\b|\bas well as\b", text[position:sentence_end], re.I):
        after = text[position + match.end():sentence_end]
        if LAST_CONJUNCT_EXEMPT.match(after):
            continue
        return True
    return False


def quality_problem(positive_question, negated_question):
    if negated_question.strip().lower() == positive_question.strip().lower():
        return "question_unchanged"
    if count_negations(negated_question) != count_negations(positive_question) + 1:
        return "negation_word_count"
    if is_head_attached(positive_question, negated_question):
        return "head_attached"
    for pattern in UNGRAMMATICAL:
        if pattern.search(negated_question):
            return "ungrammatical_pattern"
    if conjunction_after_edit(positive_question, negated_question):
        return "not_last_conjunct"
    return None


# ============================================================
# ENTITY GROUPING ("student Linda Smith")
# ============================================================

def entity_partner(sql, question, target, targets):
    """
    If `target` and a neighbouring `=` predicate (joined by AND) together
    name one entity in the question, return (partner, question_span).
    """
    if target["op"] != "=":
        return None
    for other in targets:
        if other is target or other["op"] != "=":
            continue
        first, second = sorted([target, other], key=lambda o: o["start"])
        _, first_end = predicate_span(sql, first)
        second_start, _ = predicate_span(sql, second)
        if not re.fullmatch(r"\s*AND\s*", sql[first_end:second_start], re.I):
            continue
        a, b = literal_text(first), literal_text(second)
        if not a or not b:
            continue
        for phrase in (f"{a} {b}", f"{b} {a}"):
            match = re.search(r"(?<!\w)%s(?!\w)" % re.escape(phrase), question, re.I)
            if match:
                return other, match.span()
    return None


def negate_entity(sql, target, partner):
    first, second = sorted([target, partner], key=lambda o: o["start"])
    start, _ = predicate_span(sql, first)
    _, end = predicate_span(sql, second)
    replacement = "NOT ({} = {} AND {} = {})".format(
        first["lhs"], first["rhs"], second["lhs"], second["rhs"]
    )
    return sql[:start] + replacement + sql[end:]


# ============================================================
# DATABASE
# ============================================================

def find_database(db_id):
    folder = os.path.join(DATABASE_DIR, db_id)
    standard = os.path.join(folder, f"{db_id}.sqlite")
    if os.path.exists(standard):
        return standard
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            if filename.endswith(".sqlite"):
                return os.path.join(folder, filename)
    return None


def execute_query(db_path, sql):
    connection = None
    try:
        connection = sqlite3.connect(db_path)
        connection.text_factory = lambda b: b.decode("utf-8", "ignore")
        started = time.time()
        connection.set_progress_handler(
            lambda: 1 if time.time() - started > QUERY_TIMEOUT_SECONDS else 0, 10000
        )
        rows = connection.execute(sql).fetchall()
        return True, [[None if v is None else str(v) for v in row] for row in rows], None
    except Exception as error:
        return False, None, str(error)
    finally:
        if connection is not None:
            connection.close()


def normalize_answer(answer):
    return sorted(
        tuple("" if v is None else str(v).strip().lower() for v in row)
        for row in answer
    )


# ============================================================
# PROCESS ONE DATASET
# ============================================================

def process_dataset(dataset_name, input_file):
    print("\n" + "=" * 70)
    print(f"GENERATING NEGATION CANDIDATES: {dataset_name}")
    print("=" * 70)

    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        return None

    with open(input_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    print(f"Loaded {len(data)} canonical positive examples")

    candidates = []
    q_stats = collections.Counter()      # one count per question
    p_stats = collections.Counter()      # one count per predicate
    rules = collections.Counter()
    types = collections.Counter()

    for example in tqdm(data, desc=dataset_name):
        q_stats["input_questions"] += 1
        example_id = example["id"]
        db_id = example["db_id"]
        positive_question = example["positive_question"]
        positive_sql = example["positive_sql"]
        positive_answer = example["positive_answer"]

        db_path = find_database(db_id)
        if db_path is None:
            q_stats["database_missing"] += 1
            continue

        targets = negatable_targets(find_operators(positive_sql))
        if not targets:
            q_stats["no_value_predicate"] += 1
            continue

        produced = 0
        handled_entities = set()

        for index, target in enumerate(targets):
            p_stats["predicates_found"] += 1

            partner = entity_partner(positive_sql, positive_question, target, targets)
            if partner is not None:
                other, span = partner
                key = tuple(sorted((target["start"], other["start"])))
                if key in handled_entities:
                    p_stats["skipped_entity_partner"] += 1
                    continue
                handled_entities.add(key)
                negated_sql = negate_entity(positive_sql, target, other)
                negation_type = "entity_to_not_entity"
                negated_question, rule = rewrite_question(positive_question, target, span)
                predicate_text = "NOT ({} = {} AND {} = {})".format(
                    target["lhs"], target["rhs"], other["lhs"], other["rhs"])
            else:
                negated_sql = negate_operator(positive_sql, target)
                _, negation_type = NEGATION[target["op"]]
                negated_question, rule = rewrite_question(
                    positive_question, target,
                    subquery_targets=sum(1 for t in targets if t["rhs"].startswith("(")
                                         and t["op"] in ("IN", "=")),
                )
                predicate_text = "{} {} {}".format(target["lhs"], target["op"], target["rhs"])

            if negated_question is None:
                p_stats[f"rejected:{rule}"] += 1
                continue

            problem = quality_problem(positive_question, negated_question)
            if problem:
                p_stats[f"rejected:quality_{problem}"] += 1
                continue

            ok, negated_answer, error = execute_query(db_path, negated_sql)
            if not ok:
                p_stats["rejected:sql_execution_failed"] += 1
                continue

            full_count = len(negated_answer)
            identical = normalize_answer(negated_answer) == normalize_answer(positive_answer)
            truncated = full_count > MAX_STORED_ROWS

            candidates.append({
                "candidate_id": f"{example_id}_neg_{index}",
                "source_id": example_id,
                "source_dataset": dataset_name,
                "db_id": db_id,
                "positive_question": positive_question,
                "positive_sql": positive_sql,
                "positive_answer": positive_answer,
                "positive_answer_count": len(positive_answer),
                "negated_question": negated_question,
                "negated_sql": negated_sql,
                "negated_answer": negated_answer[:MAX_STORED_ROWS],
                "negated_answer_count": full_count,
                "negated_answer_truncated": truncated,
                "answers_identical": identical,
                "negation_type": negation_type,
                "question_rule": rule,
                "predicate": predicate_text,
                "operator_original": target["op"],
                "operator_position": index,
            })
            p_stats["candidates_generated"] += 1
            rules[rule] += 1
            types[negation_type] += 1
            produced += 1

        if produced:
            q_stats["questions_with_candidate"] += 1
        else:
            q_stats["questions_all_predicates_failed"] += 1

    output_file = os.path.join(OUTPUT_DIR, f"negated_candidates_{dataset_name}.json")
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(candidates, handle, indent=2, ensure_ascii=False)

    print(f"\nRESULTS: {dataset_name}")
    print("-" * 70)
    for key in sorted(q_stats):
        print(f"  [question]  {key:<44} {q_stats[key]:>6}")
    for key in sorted(p_stats):
        print(f"  [predicate] {key:<44} {p_stats[key]:>6}")
    print(f"\nSaved to: {output_file}")

    return {
        "questions": dict(q_stats),
        "predicates": dict(p_stats),
        "question_rules": dict(rules),
        "negation_types": dict(types),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("STAGE 2 (v2): GENERATING NEGATED QA CANDIDATES")
    print("=" * 70)

    report = {}
    for dataset_name, input_file in INPUT_DATASETS.items():
        result = process_dataset(dataset_name, input_file)
        if result is not None:
            report[dataset_name] = result

    total_in = sum(r["questions"].get("input_questions", 0) for r in report.values())
    total_q = sum(r["questions"].get("questions_with_candidate", 0) for r in report.values())
    total_c = sum(r["predicates"].get("candidates_generated", 0) for r in report.values())
    report["_summary"] = {
        "input_questions": total_in,
        "questions_with_candidate": total_q,
        "candidates": total_c,
        "kept_percent": round(100 * total_q / total_in, 2) if total_in else 0,
    }
    with open(REPORT_FILE, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\n" + "=" * 70)
    print(f"INPUT QUESTIONS: {total_in}   WITH CANDIDATE: {total_q} "
          f"({report['_summary']['kept_percent']}%)   CANDIDATES: {total_c}")
    print(f"Report: {REPORT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()