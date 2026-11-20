#!/usr/bin/env python3
"""
Build negated (question, SPARQL) pairs for the CWQ *comparative* subset.

Every row in this subset carries exactly one comparative constraint, expressed
in SPARQL as a FILTER over ?num, e.g.

    FILTER (STR(?num) > "2001-11-27")      # date constraint
    FILTER (xsd:integer(?num) < 2)         # numeric constraint

That FILTER is the ground truth for what has to be negated, so the whole
pipeline is anchored on it rather than on guessing from the English alone.

Two rules:

  NORMAL  -- the comparative constraint is the last clause of the question.
             Negate it: insert "not" before the comparative phrase, and flip
             the SPARQL operator ( > <-> <= , < <-> >= ).

  EXCEPTION -- the comparative constraint appears *before* a conjunction
             ("and", "but", "also", "while", "as well as"). The clause after
             the conjunction is negated instead. The ?num FILTER is left
             untouched and the rest of the graph pattern is wrapped in
             FILTER NOT EXISTS { ... }.

The exception applies to date *and* numeric constraints alike.
"""

import os
import re
import sys

import pandas as pd
import spacy

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(BASE_DIR, "final_datasets", "qa_dataset_comparative.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "qa_dataset_comparative_exception_fixed.csv")

# The source file is UTF-8 with a BOM. Reading it as latin1 (as an earlier
# revision did) turns accented names into mojibake, so decode it properly.
INPUT_ENCODING = "utf-8-sig"
OUTPUT_ENCODING = "utf-8-sig"


print("Loading spaCy model...")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise SystemExit(
        "\nspaCy model 'en_core_web_sm' is not installed.\n"
        "Install it using:\n"
        "    python -m spacy download en_core_web_sm\n"
    )


# ============================================================
# SPARQL: LOCATING AND READING THE COMPARATIVE FILTER
# ============================================================

# ?num, optionally wrapped in STR()/xsd:integer() closing parens, followed by a
# comparison operator. This is what distinguishes the comparative FILTER from
# the many other FILTERs in these queries (?sk0 time qualifiers, ?x != guards).
NUM_OP_RE = re.compile(r"(\?num\s*\)*\s*)(>=|<=|>|<)")

FLIP_OP = {">": "<=", "<": ">=", ">=": "<", "<=": ">"}


def find_num_filter_index(lines):
    """Index of the line holding the comparative FILTER over ?num."""
    for i, line in enumerate(lines):
        if "FILTER" in line.upper() and NUM_OP_RE.search(line):
            return i
    return None


def read_constraint(sparql):
    """
    Return (kind, operator, literal) for the comparative FILTER.

    kind is "date" or "int"; literal is the raw string / number text.
    """
    lines = str(sparql).splitlines()
    idx = find_num_filter_index(lines)
    if idx is None:
        return None

    line = lines[idx]
    op = NUM_OP_RE.search(line).group(2)

    m = re.search(r'"([^"]*)"', line)
    if m:
        return {"kind": "date", "op": op, "literal": m.group(1), "line": idx}

    m = re.search(NUM_OP_RE.pattern + r"\s*(-?\d+)", line)
    if m:
        return {"kind": "int", "op": op, "literal": m.group(3), "line": idx}

    return None


def flip_num_filter(line):
    return NUM_OP_RE.sub(
        lambda m: m.group(1) + FLIP_OP[m.group(2)],
        line,
        count=1,
    )


def negate_sparql_normal(sparql):
    """Flip the comparative operator. Returns (query, ok)."""
    lines = str(sparql).splitlines()
    idx = find_num_filter_index(lines)
    if idx is None:
        return sparql, False
    lines[idx] = flip_num_filter(lines[idx])
    return "\n".join(lines), True


# ============================================================
# SPARQL: EXCEPTION RESTRUCTURE
# ============================================================

TRIPLE_RE = re.compile(r"^\s*(\S+)\s+(\S+)\s+(\S+)\s*\.\s*$")
GUARD_RE = re.compile(r"^\s*FILTER\s*\(\s*\?x\s*!=", re.IGNORECASE)

# Predicates that restrict the *type* of the answer variable. These come from
# the question's head noun ("What country ...", "What movie ..."), not from the
# clause being negated, so they stay in the main graph pattern -- otherwise
# "and does not border Germany" would also admit things that are not countries.
TYPE_PREDICATES = {
    "ns:common.topic.notable_types",
    "ns:type.object.type",
    "ns:kg.object_profile.prominent_type",
    "ns:base.biblioness.bibs_location.loc_type",
}


def _balanced(line):
    return line.count("(") == line.count(")") and line.count("{") == line.count("}")


def _vars(text):
    return set(re.findall(r"\?[A-Za-z_][A-Za-z0-9_]*", text))


def find_where_body(lines):
    """
    Return (body_start, body_end) as an exclusive-end slice of the WHERE body.

    body_start is the line after the one carrying the WHERE opening brace;
    body_end is the index of the final closing brace.
    """
    start = None
    for i, line in enumerate(lines):
        if re.search(r"\bWHERE\b", line, re.IGNORECASE):
            if "{" in line:
                start = i + 1
            elif i + 1 < len(lines) and lines[i + 1].strip().startswith("{"):
                start = i + 2
            break
    if start is None:
        return None

    end = None
    for i in range(len(lines) - 1, start - 1, -1):
        if lines[i].strip() == "}":
            end = i
            break
    if end is None or end <= start:
        return None

    return start, end


def find_comparative_branch(lines, filter_idx):
    """
    Walk backwards from the ?num FILTER, collecting the triples that bind ?num.

    Two shapes occur in this dataset:

        ?x <pred> ?num .                       (single hop)
        ?x <pred> ?c .  ?c <pred2> ?num .      (via a CVT node)

    Returns the sorted line indices, or None if the branch is not a clean
    chain rooted at ?x (hand-written queries, inline triples, etc.).
    """
    collected = []
    wanted = "?num"

    i = filter_idx - 1
    while i >= 0:
        m = TRIPLE_RE.match(lines[i])
        if not m:
            break
        subj, _pred, obj = m.groups()
        if obj != wanted:
            break
        collected.append(i)
        wanted = subj
        i -= 1
        if subj == "?x":
            break

    if not collected:
        return None

    root = TRIPLE_RE.match(lines[collected[-1]]).group(1)
    if root != "?x":
        return None

    return sorted(collected)


# A few hand-written queries ("#MANUAL SPARQL") glue the comparative branch
# onto the end of the preceding line, e.g. "FILTER (?x != ns:m.09b6zr)?x
# ns:government.politician.government_positions_held ?c .". Splitting at that
# boundary lets the restructure see the branch as its own triple.
GLUED_RE = re.compile(r"([)}])(\?[A-Za-z_][A-Za-z0-9_]*\s+ns:)")


def split_glued_lines(lines):
    out = []
    for line in lines:
        parts = GLUED_RE.sub(lambda m: m.group(1) + "\n" + m.group(2), line)
        out.extend(parts.split("\n"))
    return out


def negate_sparql_exception(sparql):
    """
    Keep the comparative constraint, negate everything else.

    The comparative branch is hoisted to the top of the WHERE body so it still
    binds ?x, and the remaining graph pattern is wrapped in FILTER NOT EXISTS.
    Standalone `FILTER (?x != ns:m.xxx)` guards stay at the top level.

    Returns (query, ok). ok is False when the query shape is not recognised,
    in which case the caller falls back to the normal rule.
    """
    lines = split_glued_lines(str(sparql).splitlines())

    filter_idx = find_num_filter_index(lines)
    if filter_idx is None:
        return sparql, False

    span = find_where_body(lines)
    if span is None:
        return sparql, False
    body_start, body_end = span

    if not (body_start <= filter_idx < body_end):
        return sparql, False

    branch = find_comparative_branch(lines, filter_idx)
    if branch is None or branch[0] < body_start:
        return sparql, False

    keep = set(branch) | {filter_idx}

    # Type restrictions on ?x, plus the variables they introduce, stay outside.
    type_idx = []
    type_vars = set()
    for i in range(body_start, body_end):
        if i in keep:
            continue
        m = TRIPLE_RE.match(lines[i])
        if m and m.group(1) == "?x" and m.group(2) in TYPE_PREDICATES:
            type_idx.append(i)
            type_vars |= _vars(m.group(3))

    # A self-contained FILTER over only those variables belongs with them.
    for i in range(body_start, body_end):
        if i in keep or i in type_idx:
            continue
        line = lines[i]
        if "FILTER" not in line.upper() or not _balanced(line):
            continue
        refs = _vars(line)
        if refs and refs <= type_vars:
            type_idx.append(i)

    keep_outside = set(type_idx)

    guards = []
    rest = []
    for i in range(body_start, body_end):
        if i in keep or i in keep_outside:
            continue
        line = lines[i]
        if not line.strip():
            continue
        if GUARD_RE.match(line):
            guards.append(line.rstrip())
        else:
            rest.append(line.rstrip())

    # Nothing left to negate -- the whole query *is* the comparative branch.
    if not rest:
        return sparql, False

    out = lines[:body_start]
    out.extend(guards)
    out.extend(lines[i].rstrip() for i in sorted(keep_outside))
    out.extend(lines[i].rstrip() for i in branch)
    out.append(lines[filter_idx].rstrip())
    out.append("FILTER NOT EXISTS {")
    out.extend("  " + line for line in rest)
    out.append("}")
    out.extend(lines[body_end:])

    return "\n".join(out), True


# ============================================================
# LOCATING THE CONSTRAINT INSIDE THE QUESTION
# ============================================================

MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

# Month number -> alternation of the spellings that occur in CWQ.
MONTH_ALT = {}
for _name, _num in MONTHS.items():
    MONTH_ALT.setdefault(_num, []).append(_name)
MONTH_ALT = {
    _num: "|".join(sorted(_names, key=len, reverse=True))
    for _num, _names in MONTH_ALT.items()
}

# Numbers written as words, so "ran for more than two seasons" is matched to
# FILTER (xsd:integer(?num) > 2).
NUMBER_WORDS = {
    0: "zero|no", 1: "one|a|an|first", 2: "two|second", 3: "three|third",
    4: "four|fourth", 5: "five|fifth", 6: "six|sixth", 7: "seven|seventh",
    8: "eight|eighth", 9: "nine|ninth", 10: "ten|tenth",
    11: "eleven|eleventh", 12: "twelve|twelfth", 13: "thirteen|thirteenth",
    14: "fourteen|fourteenth", 15: "fifteen|fifteenth",
    16: "sixteen|sixteenth", 17: "seventeen|seventeenth",
    18: "eighteen|eighteenth", 19: "nineteen|nineteenth",
    20: "twenty|twentieth",
}


def _search(question, pattern):
    """Match `pattern` where it is not glued to surrounding digits.

    Letters may abut, because CWQ contains run-together text like "after1996".
    """
    return re.search(
        r"(?<!\d)(?:" + pattern + r")(?!\d)",
        question,
        flags=re.IGNORECASE,
    )


def locate_date(question, literal):
    """
    Find the surface form of a SPARQL date literal inside the question.

    Candidate spellings are tried most-specific first, so for the literal
    "1986-11-18" the question "... prime minister in 1993 ... before 1986-11-18?"
    matches the full date rather than the incidental year 1993.

    Years are not restricted to a century range: organisation founding dates in
    this subset go back to 1179.
    """
    parts = [int(p) for p in re.findall(r"\d+", literal)]
    if not parts:
        return None

    year = parts[0]
    month = parts[1] if len(parts) > 1 else None
    day = parts[2] if len(parts) > 2 else None

    y4 = str(year)
    y2 = "%02d" % (year % 100)

    candidates = []

    if month and day and month in MONTH_ALT:
        mn = MONTH_ALT[month]
        candidates += [
            # November 27, 2001 / Nov 27 2001 / October 30 of 2011
            rf"(?:{mn})\.?\s+0?{day}(?:st|nd|rd|th)?(?:\s*,|\s+of)?\s+{y4}",
            # 27 November 2001 / 28th of April, 1603
            rf"0?{day}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{mn})\.?(?:\s*,)?\s+{y4}",
        ]
    if month and day:
        candidates += [
            rf"{y4}[-/]0?{month}[-/]0?{day}",
            rf"0?{month}[-/]0?{day}[-/](?:{y4}|{y2})",
            rf"0?{day}[-/]0?{month}[-/](?:{y4}|{y2})",
        ]
    if month:
        if month in MONTH_ALT:
            candidates.append(rf"(?:{MONTH_ALT[month]})\.?\s+{y4}")
        candidates.append(rf"{y4}[-/]0?{month}")

    candidates.append(y4)

    for pattern in candidates:
        m = _search(question, pattern)
        if m:
            return m

    return None


def locate_int(question, literal):
    """
    Find a numeric literal in the question.

    Tolerates comma/space digit grouping ("987,007"), zero padding in the
    SPARQL literal ("056" vs "56") and small numbers written as words or
    ordinals ("two seasons", "the first Vice President").
    """
    negative = literal.startswith("-")
    digits = literal.lstrip("-")

    forms = []
    for variant in (digits, digits.lstrip("0") or "0"):
        grouped = r"[,\s]?".join(re.escape(d) for d in variant)
        forms.append((r"-?" if negative else r"-?") + grouped)

    value = int(digits.lstrip("0") or "0")
    if not negative and value in NUMBER_WORDS:
        forms.append(r"(?:" + NUMBER_WORDS[value] + r")(?![A-Za-z])")

    for form in forms:
        m = _search(question, form)
        if m:
            return m

    return None


def _levenshtein(a, b):
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def locate_date_relaxed(question, literal):
    """
    Approximate fallback for dates.

    The source data contains transcription slips where the question's year does
    not match the SPARQL literal ("opened before May 30, 1030" against
    1930-05-30, "released after 09-23-2009" against 2000-09-23). The month and
    day still identify the constraint, and the comparative phrase in front of it
    is exactly what has to be negated, so a near miss is still usable.
    """
    parts = [int(p) for p in re.findall(r"\d+", literal)]
    if not parts:
        return None
    year = parts[0]
    month = parts[1] if len(parts) > 1 else None
    day = parts[2] if len(parts) > 2 else None

    any_year = r"\d{3,4}"

    if month and day:
        candidates = []
        if month in MONTH_ALT:
            mn = MONTH_ALT[month]
            candidates += [
                rf"(?:{mn})\.?\s+0?{day}(?:st|nd|rd|th)?(?:\s*,|\s+of)?\s+{any_year}",
                rf"0?{day}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{mn})\.?(?:\s*,)?\s+{any_year}",
            ]
        candidates += [
            rf"{any_year}[-/]0?{month}[-/]0?{day}",
            rf"0?{month}[-/]0?{day}[-/]{any_year}",
            rf"0?{day}[-/]0?{month}[-/]{any_year}",
        ]
        for pattern in candidates:
            m = _search(question, pattern)
            if m:
                return m

    # Otherwise accept a year that is within a couple of typos of the literal.
    best, best_dist = None, 3
    for m in re.finditer(r"(?<!\d)\d{3,4}(?!\d)", question):
        dist = _levenshtein(m.group(), str(year))
        if dist < best_dist:
            best, best_dist = m, dist
    return best


def locate_int_relaxed(question, literal):
    """Approximate fallback for numbers, for the same transcription slips."""
    target = literal.lstrip("-").lstrip("0") or "0"

    best, best_dist = None, 3
    for m in re.finditer(r"(?<![\d,])-?\d[\d,\s]*\d|(?<![\d,])-?\d(?![\d,])", question):
        found = re.sub(r"[^\d]", "", m.group()).lstrip("0") or "0"
        if abs(len(found) - len(target)) > 1:
            continue
        dist = _levenshtein(found, target)
        if dist < best_dist:
            best, best_dist = m, dist
    return best


def locate_constraint(question, constraint):
    """Returns (match, quality) where quality is "exact", "approximate" or ""."""
    if constraint["kind"] == "date":
        exact, relaxed = locate_date, locate_date_relaxed
    else:
        exact, relaxed = locate_int, locate_int_relaxed

    m = exact(question, constraint["literal"])
    if m:
        return m, "exact"

    m = relaxed(question, constraint["literal"])
    if m:
        return m, "approximate"

    return None, ""


# ============================================================
# COMPARATIVE PHRASES
# ============================================================

# Head words of a comparative / temporal phrase, searched in a short window
# immediately before the literal. Misspellings that occur in CWQ are included
# ("less then", "more that", "pror to", "bofre", "lafter", "smalelr").
# Head words of a comparative / temporal phrase, searched in a short window
# immediately before the literal. Misspellings that occur in CWQ are included
# ("less then", "more that", "pror to", "bofre", "lafter", "smalelr").
#
# STRONG heads carry the comparison itself. WEAK heads are bare prepositions
# that only carry one when nothing stronger is present -- in "number of
# postgraduates greater than 1686" the word to negate is "greater", not "of".
STRONG_PHRASES = [
    r"no\s+(?:more|less|fewer|earlier|later)",
    r"at\s+(?:least|most)",
    r"as\s+of",
    r"prior\s+to", r"pror\s+to", r"prior", r"pror",
    r"greater", r"smaller", r"smalelr", r"larger", r"higher", r"lower",
    r"bigger", r"longer", r"shorter", r"fewer", r"few",
    r"more", r"less", r"over", r"under", r"above", r"below", r"beneath",
    r"exceed(?:s|ing|ed)?", r"beyond", r"upwards\s+of", r"up\s+to",
    r"lafter", r"later", r"earlier", r"bofre", r"before", r"after",
    r"since", r"until", r"till", r"from", r"post", r"pre",
    r"beginning", r"ending", r"starting", r"during", r"between",
]

WEAK_PHRASES = [r"by", r"on", r"in", r"of", r"around", r"circa"]


def _phrase_re(phrases):
    return re.compile(
        r"(?<![A-Za-z])(?:" + "|".join(phrases) + r")(?![A-Za-z])",
        flags=re.IGNORECASE,
    )


STRONG_RE = _phrase_re(STRONG_PHRASES)
WEAK_RE = _phrase_re(WEAK_PHRASES)

# What may sit between a weak preposition and the literal it introduces:
# whitespace, an article, a month name, or another weak preposition, so that
# "played in January of 2013" attaches to "in" rather than to "of".
WEAK_GAP_RE = re.compile(
    r"[\s,]*(?:(?:the|a|an|" + "|".join(MONTHS) + r"|"
    + "|".join(WEAK_PHRASES) + r")[\s,.]*)*",
    flags=re.IGNORECASE,
)

# Comparatives that trail the literal ("8993985 or less", "28 plus").
TRAILING_RE = re.compile(
    r"^\s*(?:or\s+(?:more|less|fewer|greater|higher|lower|earlier|later)|plus|"
    r"or\s+above|or\s+below)(?![A-Za-z])",
    flags=re.IGNORECASE,
)

WINDOW = 45


def find_comparative_phrase(question, lit_start, lit_end):
    """
    Locate the phrase that carries the comparison.

    Returns (start, end, position) where position is "leading" or "trailing",
    or None when the question states the constraint with no comparative wording
    at all (e.g. "Which TV show with the tvrage_id 4914 ...").
    """
    if TRAILING_RE.match(question[lit_end:]):
        return lit_start, lit_start, "trailing"

    window_start = max(0, lit_start - WINDOW)
    window = question[window_start:lit_start]

    # The nearest strong head governs the literal.
    strong = list(STRONG_RE.finditer(window))
    if strong:
        chosen = strong[-1]
        return window_start + chosen.start(), window_start + chosen.end(), "leading"

    # Otherwise a bare preposition may still carry the constraint ("founded in
    # 1837"), but only when it actually leads into the literal. Requiring that
    # keeps "whose date of death is March 11, 1984" from negating the "of" in
    # "date of death", which has nothing to do with the comparison.
    for m in WEAK_RE.finditer(window):
        gap = window[m.end():]
        if WEAK_GAP_RE.fullmatch(gap):
            return window_start + m.start(), window_start + m.end(), "leading"

    return None


def negate_comparative_phrase(question, phrase_start):
    """Insert "not" before the comparative phrase, matching the dataset style."""
    return question[:phrase_start] + "not " + question[phrase_start:]


# ============================================================
# CONJUNCTIONS
# ============================================================

CONJ_RE = re.compile(
    r"(?<![A-Za-z])(?:and|but|also|while|as\s+well\s+as|which\s+was\s+also|ad)"
    r"(?![A-Za-z])",
    flags=re.IGNORECASE,
)


def find_conjunction_after(question, pos):
    """First clause-joining conjunction occurring after character offset pos."""
    for m in CONJ_RE.finditer(question, pos):
        # "8993985 or less" style trailing comparatives are not clause joins.
        if TRAILING_RE.match(question[m.start():]):
            continue
        return m
    return None


# ============================================================
# CLAUSE NEGATION (EXCEPTION RULE)
# ============================================================

AUX_WORDS = {
    "is", "are", "was", "were", "am", "be", "been", "being",
    "has", "have", "had",
    "does", "do", "did",
    "can", "could", "will", "would", "shall", "should",
    "may", "might", "must",
    "'s", "'re", "'ve",
}

CONTRACTED = {"can": "cannot", "will": "will not", "shall": "shall not"}


PLURAL_TAGS = {"NNS", "NNPS"}


def _question_subject_is_plural(question):
    """
    Number of the question's head noun, used when the clause after the
    conjunction has no subject of its own ("What film was released before
    2011-12-01 and feature Taylor Lautner?").
    """
    doc = nlp(question)
    for chunk in doc.noun_chunks:
        if chunk.root.pos_ in {"NOUN", "PROPN"}:
            return chunk.root.tag_ in PLURAL_TAGS
    return False


def _subject_is_plural(question, verb):
    """Number of the verb's own subject, falling back to the question's."""
    for child in verb.children:
        if child.dep_ in {"nsubj", "nsubjpass"}:
            if child.tag_ in PLURAL_TAGS:
                return True
            if child.pos_ in {"NOUN", "PROPN"}:
                return False
    return _question_subject_is_plural(question)


def _find_clause_verb(clause):
    """
    First finite verb of the clause, with its containing doc and char offset.

    The clause is parsed as-is first. When that finds no verb -- spaCy tags the
    head word of "rules Denmark right now" as a plural noun -- it is reparsed
    with a dummy subject, which supplies the missing syntactic context.
    """
    doc = nlp(clause)
    verb = next((t for t in doc if t.pos_ in {"AUX", "VERB"}), None)
    if verb is not None:
        return doc, verb, 0

    prefix = "He "
    doc = nlp(prefix + clause)
    verb = next((t for t in doc[1:] if t.pos_ in {"AUX", "VERB"}), None)
    if verb is not None:
        return doc, verb, len(prefix)

    return doc, None, len(prefix)


def negate_clause_after_conjunction(question, conj_end):
    """
    Negate the first finite verb of the clause following the conjunction.

    Returns (negated_question, status).
    """
    tail = question[conj_end:]
    stripped = tail.lstrip()
    lead = len(tail) - len(stripped)

    if not stripped:
        return None, "no_clause_after_conjunction"

    doc, verb, offset = _find_clause_verb(stripped)

    for token in doc:
        if token.lower_ in {"not", "n't", "never"}:
            return None, "already_negated"

    if verb is None:
        return None, "no_verb_after_conjunction"

    # Offset of the verb inside the original question.
    v_start = conj_end + lead + (verb.idx - offset)
    v_end = v_start + len(verb.text)
    word = verb.text.lower()

    # "have"/"do" are auxiliaries only when spaCy tags them as such. As main
    # verbs they take do-support: "and have Spanish as their primary language"
    # negates to "do not have Spanish", not "have not Spanish". Forms of "be"
    # always take the bare "not", copula or not.
    is_auxiliary = (
        verb.lemma_.lower() == "be"
        or (word in AUX_WORDS and verb.pos_ == "AUX")
    )

    if is_auxiliary:
        replacement = CONTRACTED.get(word, verb.text + " not")
    elif verb.tag_ == "VBD":
        replacement = "did not " + verb.lemma_
    elif verb.tag_ in {"VBZ", "VBP", "VB"}:
        # CWQ questions are frequently ungrammatical ("What film ... and
        # feature Taylor Lautner"), so agreement comes from the subject rather
        # than from the verb form alone.
        plural = _subject_is_plural(question, verb)
        replacement = ("do not " if plural else "does not ") + verb.lemma_
    else:
        # Participle / gerund with no auxiliary in front of it.
        replacement = "not " + verb.text

    negated = question[:v_start] + replacement + question[v_end:]
    return negated, "ok"


# ============================================================
# PER-ROW PIPELINE
# ============================================================

def process_row(question, sparql):
    """
    Returns a dict with the negated question, negated SPARQL and diagnostics.
    """
    question = str(question)
    sparql = str(sparql)

    result = {
        "negated_question": question,
        "negated_sparql": sparql,
        "negation_rule": "",
        "constraint_kind": "",
        "detected_constraint": "",
        "detected_phrase": "",
        "detected_conjunction": "",
        "question_negated": False,
        "sparql_negated": False,
        "negation_status": "",
        "literal_match": "",
    }

    constraint = read_constraint(sparql)
    if constraint is None:
        result["negation_rule"] = "none"
        result["negation_status"] = "no_comparative_filter"
        return result

    result["constraint_kind"] = constraint["kind"]
    result["detected_constraint"] = constraint["literal"]

    located, match_quality = locate_constraint(question, constraint)
    result["literal_match"] = match_quality

    # ------------------------------------------------------------------
    # Decide which rule applies: the exception needs the constraint to sit
    # before a conjunction that introduces a separate, negatable clause.
    # ------------------------------------------------------------------
    conj = None
    fallback = ""
    if located is not None:
        conj = find_conjunction_after(question, located.end())

    if conj is not None:
        negated_q, status = negate_clause_after_conjunction(question, conj.end())
        if status == "ok":
            negated_s, ok = negate_sparql_exception(sparql)
            if not ok:
                # The query shape is not one the restructure understands
                # (hand-written queries with inline triples or sub-SELECTs).
                # Fall back to the normal rule so the pair stays consistent,
                # but record it -- the exception reading was the intended one.
                fallback = "exception_fallback_sparql_shape"
            if ok:
                result.update(
                    negated_question=negated_q,
                    negated_sparql=negated_s,
                    negation_rule="conjunction_after_comparative",
                    detected_conjunction=conj.group(),
                    question_negated=True,
                    sparql_negated=True,
                    negation_status="ok",
                )
                return result
            # SPARQL shape not recognised -- fall through to the normal rule so
            # the pair stays consistent rather than half-negated.

    # ------------------------------------------------------------------
    # Normal rule
    # ------------------------------------------------------------------
    result["negation_rule"] = "normal_comparative"
    if fallback:
        result["negation_rule"] = "normal_comparative (exception fallback)"

    negated_s, sparql_ok = negate_sparql_normal(sparql)
    result["negated_sparql"] = negated_s
    result["sparql_negated"] = sparql_ok

    if located is None:
        result["negation_status"] = "constraint_not_found_in_question"
        return result

    phrase = find_comparative_phrase(question, located.start(), located.end())
    if phrase is None:
        result["negation_status"] = "no_comparative_phrase"
        return result

    p_start, p_end, position = phrase
    result["detected_phrase"] = (
        question[p_start:p_end] if position == "leading" else "(trailing)"
    )

    if re.search(r"\bnot\b", question[max(0, p_start - 12):p_start], re.IGNORECASE):
        result["negation_status"] = "already_negated"
        return result

    result["negated_question"] = negate_comparative_phrase(question, p_start)
    result["question_negated"] = True
    result["negation_status"] = fallback or "ok"
    return result


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Reading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, encoding=INPUT_ENCODING)
    print(f"Rows: {len(df)}")

    records = []
    for idx, row in df.iterrows():
        try:
            records.append(process_row(row["original_question"], row["sparql"]))
        except Exception as exc:  # keep going; the row is flagged for review
            print(f"[ERROR] row={idx} id={row.get('id')}: {exc}", file=sys.stderr)
            records.append({
                "negated_question": str(row["original_question"]),
                "negated_sparql": str(row["sparql"]),
                "negation_rule": "error",
                "constraint_kind": "",
                "detected_constraint": "",
                "detected_phrase": "",
                "detected_conjunction": "",
                "question_negated": False,
                "sparql_negated": False,
                "negation_status": f"error: {exc}",
                "literal_match": "",
            })

    out = pd.DataFrame(records, index=df.index)
    for col in out.columns:
        df[col] = out[col]

    df.to_csv(OUTPUT_FILE, index=False, encoding=OUTPUT_ENCODING)

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Input rows:            {len(df)}")
    print(f"Questions negated:     {int(df['question_negated'].sum())}")
    print(f"SPARQL negated:        {int(df['sparql_negated'].sum())}")
    print()
    print("By rule:")
    for rule, n in df["negation_rule"].value_counts().items():
        print(f"  {rule:34s} {n}")
    print()
    print("By status:")
    for status, n in df["negation_status"].value_counts().items():
        print(f"  {status:34s} {n}")

    needs_review = df[~df["question_negated"].astype(bool)]
    if len(needs_review):
        review_path = os.path.join(BASE_DIR, "needs_review.csv")
        needs_review.to_csv(review_path, index=False, encoding=OUTPUT_ENCODING)
        print(f"\n{len(needs_review)} row(s) need manual review -> {review_path}")

    print("\nOutput:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
