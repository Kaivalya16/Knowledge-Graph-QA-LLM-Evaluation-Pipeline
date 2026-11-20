#!/usr/bin/env python3
"""
Build negated (question, SPARQL) pairs for the CWQ *composition* subset.

Each question here chains two steps through an intermediate entity ?c:

    What year did [the team with Baltimore Fight Song] [win the Superbowl]
                   \____ INNER: defines ?c ____/  \___ OUTER: ?c -> ?x __/

    ?c ns:sports.sports_team.fight_song ns:m.0hn47qp .      INNER
    ?c ns:sports.sports_team.championships ?x .             OUTER path
    ?x ns:sports.sports_championship_event.championship ns:m.06x5s .   binds ?x

RULE (word order): negate the LAST phrase or clause of the question, with an
explicit "not" / "n't" (or "without"), and the SPARQL step it expresses.

  * The question ends with the INNER phrase -- the link nearest the topic
    entity is negated, and the ?c-defining triples are wrapped in
    FILTER NOT EXISTS:

        the team owned by Jerry Jones        -> the team not owned by ...
        the country where Tsonga is spoken   -> ... where Tsonga isn't spoken
        the author of The New Atlantis       -> an author that is not the
                                                author of The New Atlantis

    When the negated word only renames the entity, the relation is kept and
    the entity excluded instead:

        the concert tour called Demi World Tour -> ... not called Demi World Tour
        ?c ns:music.artist.concert_tours ?neg_e . FILTER (?neg_e != ns:m.010pzj9k)

  * The question ends with the OUTER clause -- that verb is negated and the
    ?c -> ?x path is wrapped instead:

        Where did the artist nominated for X go to school
     -> Where did the artist nominated for X not go to school

    A range triple (?x_range <last path predicate> ?x) keeps ?x bound to the
    things the step could return, unless another constraint already binds it.

The topic entity is located through labels induced from the dataset itself (a
MID appearing in several questions is named in each of them), falling back to
the capitalised phrases of the question.
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

INPUT_FILE = os.path.join(BASE_DIR, "qa_dataset_composition.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "qa_dataset_composition_fixed.csv")
REVIEW_FILE = os.path.join(BASE_DIR, "needs_review.csv")

INPUT_ENCODING = "utf-8-sig"
OUTPUT_ENCODING = "utf-8-sig"


MOJIBAKE_RE = re.compile(r"[ÃÂâ][\x80-\xbf\u0080-\u00bf]")


def fix_mojibake(text):
    """
    Repair fields that were written as UTF-8 but read as latin1 at some earlier
    stage ("SetÃºbal" -> "Setúbal"). Some rows in this file are affected and
    some are not, so the repair is applied only when it round-trips cleanly.
    """
    if not isinstance(text, str) or not MOJIBAKE_RE.search(text):
        return text
    try:
        repaired = text.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    return repaired if not MOJIBAKE_RE.search(repaired) else text


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
# SPARQL PARSING
# ============================================================

TRIPLE_RE = re.compile(r"^\s*(\S+)\s+(\S+)\s+(.+?)\s*\.\s*$")
GUARD_RE = re.compile(r"^\s*FILTER\s*\(\s*\?x\s*!=", re.IGNORECASE)

# Restrictions on the *type* of the answer, coming from the question's head noun
# ("What country ...", "What movie ..."). They belong to neither clause and
# always stay in the main pattern.
TYPE_PREDICATES = {
    "ns:common.topic.notable_types",
    "ns:type.object.type",
    "ns:base.biblioness.bibs_location.loc_type",
    "ns:kg.object_profile.prominent_type",
}

# Constructs the line-based reader cannot safely rearrange. ORDER BY / LIMIT
# ("... last win the world series") sit after the WHERE body and are left alone,
# so they do not block the rewrite.
COMPLEX_RE = re.compile(r"\bUNION\b|\bOPTIONAL\b", re.IGNORECASE)

# A trailing comment on a triple line ("?c ns:... ?y . # Michigan").
COMMENT_RE = re.compile(r"\s+#.*$")


def find_where_body(lines):
    """(start, end) exclusive-end slice of the WHERE body, or None."""
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


def is_complex(sparql):
    """True when the query has shapes this script will not rewrite."""
    body = str(sparql)
    where = body[body.upper().find("WHERE"):] if "WHERE" in body.upper() else body
    if COMPLEX_RE.search(where):
        return True
    if ";" in where:
        return True
    # a nested SELECT inside the WHERE body
    if re.search(r"\bSELECT\b", where[5:], re.IGNORECASE):
        return True
    return False


def _balanced(text):
    return text.count("(") == text.count(")") and text.count("{") == text.count("}")


VAR_RE = re.compile(r"\?[A-Za-z_][A-Za-z0-9_]*")


def parse_body(sparql):
    """
    Read the WHERE body into units, where a unit is one or more consecutive
    lines that must move together.

    The time-qualifier filters in this dataset span several lines:

        FILTER(NOT EXISTS {?y ns:...roster.from ?sk0} ||
        EXISTS {?y ns:...roster.from ?sk1 .
        FILTER(STR(?sk1) <= "2012-12-31") })

    Reading those line by line both invents nonsense triples ("EXISTS {?y ...")
    and, if the leftover lines are dropped, leaves unbalanced braces. Units are
    therefore accumulated until their brackets balance.

    Returns None when the body cannot be read.
    """
    lines = str(sparql).splitlines()
    span = find_where_body(lines)
    if span is None:
        return None
    body_start, body_end = span

    units = []
    i = body_start
    while i < body_end:
        if not lines[i].strip():
            i += 1
            continue
        idx = [i]
        text = lines[i]
        # pull in continuation lines until the brackets close
        while not _balanced(text) and idx[-1] + 1 < body_end:
            idx.append(idx[-1] + 1)
            text += "\n" + lines[idx[-1]]
        i = idx[-1] + 1

        stripped = text.strip()
        if stripped in "{}" or stripped.startswith("#"):
            units.append({"idx": idx, "kind": "other", "vars": set()})
            continue

        if GUARD_RE.match(text):
            units.append({"idx": idx, "kind": "guard", "vars": {"?x"}})
            continue

        if "FILTER" in stripped.upper():
            units.append({
                "idx": idx,
                "kind": "filter",
                "vars": set(VAR_RE.findall(text)),
            })
            continue

        m = TRIPLE_RE.match(COMMENT_RE.sub("", text)) if len(idx) == 1 else None
        if m:
            subj, pred, obj = m.group(1), m.group(2), m.group(3).strip()
            # "?c ns:common.topic.notable_types ?x" is the outer step itself
            # ("What work does the subject of the movie Boadicea do"), not a
            # restriction on the answer type.
            kind = "type" if pred in TYPE_PREDICATES and not obj.startswith("?") else "triple"
            units.append({
                "idx": idx, "kind": kind, "s": subj, "p": pred, "o": obj,
                "vars": set(VAR_RE.findall(text)),
            })
            continue

        units.append({"idx": idx, "kind": "other", "vars": set(VAR_RE.findall(text))})

    return {"lines": lines, "body": (body_start, body_end), "units": units}


def split_composition(units):
    """
    Split the triples into the two steps of the composition.

    Returns (path, inner, xonly):

      path   the ?c -> ?x chain, i.e. the triples the question's matrix clause
             expresses ("... win the Superbowl")
      inner  the branch that defines ?c ("the team with Baltimore Fight Song")
      xonly  constraints hanging off ?x alone; these are what still binds ?x if
             the path is negated
    """
    triples = [u for u in units if u["kind"] == "triple"]
    if not triples:
        return [], [], []

    def reaches_x(t, seen):
        if t["o"] == "?x":
            return True
        if not t["o"].startswith("?"):
            return False
        for u in triples:
            if id(u) in seen or u["s"] != t["o"]:
                continue
            if reaches_x(u, seen | {id(u)}):
                return True
        return False

    path = {id(t) for t in triples if t["s"] == "?c" and reaches_x(t, {id(t)})}
    grew = True
    while grew:
        grew = False
        objects = {t["o"] for t in triples if id(t) in path}
        for t in triples:
            if id(t) in path or t["s"] not in objects:
                continue
            if t["o"] == "?x" or reaches_x(t, {id(t)}):
                path.add(id(t))
                grew = True

    inner = [t for t in triples if id(t) not in path and t["s"] != "?x"]
    xonly = [t for t in triples if id(t) not in path and t["s"] == "?x"]
    return [t for t in triples if id(t) in path], inner, xonly


# ============================================================
# ENTITY LABELS, INDUCED FROM THE DATASET
# ============================================================

MID_RE = re.compile(r"ns:((?:m|g)\.[A-Za-z0-9_]+)")

# A capitalised phrase, allowing lowercase joiners inside it ("Palace of Knossos").
CAP_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9'\u2019]*"
    r"(?:\s+(?:of|the|de|del|da|van|von|and|for)\s+[A-Z][A-Za-z0-9'\u2019]*"
    r"|\s+[A-Z][A-Za-z0-9'\u2019]*)*"
)

QUESTION_WORDS = {
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "the", "a", "an", "of", "in", "on", "at", "is", "are", "was", "were",
    "did", "does", "do", "has", "have", "had", "name", "list", "tell", "and",
    "or", "but", "also", "both", "there", "this", "that", "these", "those",
    "for", "from", "with", "by", "to", "his", "her", "its", "their", "s",
}


ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)


def _normalise(phrase):
    """Key that merges spelling variants: "Super Bowl" and "Superbowl"."""
    return re.sub(r"[^a-z0-9]", "", ARTICLE_RE.sub("", phrase).lower())


def induce_labels(questions, sparqls, min_support=3, min_coverage=0.5):
    """
    Work out what each Freebase MID is called, using the questions themselves.

    Freebase is gone and this project has no MID -> name file, but a MID that
    appears in several questions is almost always named in each of them: every
    query mentioning ns:m.06x5s sits next to a question about the Super Bowl.
    So for each MID the capitalised phrases of its questions are pooled and the
    most widely shared one wins.

    Phrases are pooled by a spelling-insensitive key, because otherwise
    "Superbowl" and "Super Bowl" split 48 questions into two 38% shares and
    neither clears the threshold. Every surface form seen for the winning key
    is returned, so the match works whichever spelling a question uses.

    Labels are what makes the clause -> triple mapping exact rather than
    inferred, so this is the single biggest lever on SPARQL coverage.
    """
    by_mid = {}
    for q, s in zip(questions, sparqls):
        q = str(q)
        for mid in set(MID_RE.findall(str(s))):
            by_mid.setdefault(mid, []).append(q)

    scored = {}
    for mid, qs in by_mid.items():
        if len(qs) < min_support:
            continue
        counts = {}
        forms = {}
        for q in qs:
            seen = set()
            for m in CAP_RE.finditer(q):
                phrase = m.group().strip()
                if len(phrase) < 3 or phrase.lower() in QUESTION_WORDS:
                    continue
                key = _normalise(phrase)
                if not key:
                    continue
                seen.add(key)
                forms.setdefault(key, set()).add(phrase)
            for key in seen:
                counts[key] = counts.get(key, 0) + 1

        best = None
        for key, n in counts.items():
            coverage = n / len(qs)
            if coverage < min_coverage:
                continue
            rank = (round(coverage, 2), len(key))
            if best is None or rank > best[0]:
                best = (rank, key, coverage)
        if best:
            scored[mid] = (best[1], sorted(forms[best[1]]), best[2])

    # A name shared by two MIDs is only ambiguous where both can appear at
    # once. Several Super Bowl MIDs are all called "Super Bowl", and dropping
    # every one of them loses the label for questions where just one occurs.
    together = set()
    for s in sparqls:
        mids = sorted(set(MID_RE.findall(str(s))))
        for i, a in enumerate(mids):
            for b in mids[i + 1:]:
                together.add((a, b))

    owners = {}
    for mid, (key, _forms, _coverage) in scored.items():
        owners.setdefault(key, []).append(mid)

    labels = {}
    for key, mids in owners.items():
        clash = any(
            (a, b) in together
            for i, a in enumerate(sorted(mids)) for b in sorted(mids)[i + 1:]
        )
        if clash:
            continue
        for mid in mids:
            labels[mid] = scored[mid][1]
    return labels


# ============================================================
# LOCATING ENTITIES AND LITERALS IN THE QUESTION
# ============================================================

def literal_patterns(obj):
    """Regexes locating a literal object of a triple inside the question."""
    if obj.startswith(("ns:", "?", "[")):
        return []
    value = obj.strip().strip(".").strip()
    value = re.sub(r"\^\^.*$", "", value).strip('"').strip("'").strip()
    if not value:
        return []

    pats = []
    digits = re.sub(r"[^\d]", "", value)
    if digits:
        # tolerate comma/space digit grouping: 1464000 vs 1,464,000
        pats.append(r"(?<!\d)" + r"[,\s]?".join(digits) + r"(?!\d)")
    if len(value) >= 3 and re.search(r"[A-Za-z]", value):
        pats.append(re.escape(value))
    return pats


def locate_spans(question, patterns):
    """(start, end) of every match of any pattern."""
    spans = []
    for pat in patterns:
        try:
            spans += [m.span() for m in re.finditer(pat, question, re.IGNORECASE)]
        except re.error:
            continue
    return spans


def label_patterns(forms):
    """Regexes for a label's surface forms, tolerating a leading article."""
    pats = []
    for form in forms:
        bare = ARTICLE_RE.sub("", form)
        for variant in {form, bare}:
            if len(variant) >= 3:
                pats.append(r"(?:the\s+|a\s+|an\s+)?" + re.escape(variant))
    return pats


def group_offsets(question, group, labels):
    """
    Character spans where a group's constraint is named in the question -- its
    entities by their induced label, plus any literal object.

    Spans rather than start positions, so that a coordination falling *inside*
    a name can be told apart from one separating the two constraints ("the book
    The Fish the Fighters and the Song" is one title, not a coordination).
    """
    spans = []
    for t in group:
        if labels:
            for mid in MID_RE.findall(t["s"]) + MID_RE.findall(t["o"]):
                forms = labels.get(mid)
                if forms:
                    spans += locate_spans(question, label_patterns(forms))
        spans += locate_spans(question, literal_patterns(t["o"]))
    return spans


# ============================================================
# NL: CONTRACTIONS
# ============================================================

CONTRACTIONS = {
    "is": "isn't", "are": "aren't", "was": "wasn't", "were": "weren't",
    "am": "am not", "be": "not be", "been": "not been",
    "has": "hasn't", "have": "haven't", "had": "hadn't",
    "do": "don't", "does": "doesn't", "did": "didn't",
    "can": "can't", "could": "couldn't", "will": "won't",
    "would": "wouldn't", "shall": "shan't", "should": "shouldn't",
    "may": "may not", "might": "might not", "must": "mustn't",
}


# ============================================================
# NL: LOCATING THE INNER CONSTRAINT
# ============================================================

def entity_candidates(question, inner, outer, labels):
    """
    Where the inner constraint's entity is named in the question.

    Returns (candidates, source): a list of span lists, best first. When the
    induced labels cover the inner MID that span comes first. After it, every
    capitalised phrase not belonging to the outer side is a candidate, longest
    first, and the caller keeps the first one that sits in a clear modifier -- in "Who plays Jacob Black in the movie in which Krys Hyatt also
    played a part" the longer name is the outer character, and only "Krys
    Hyatt" sits in a relative clause.
    """
    own = group_offsets(question, inner, labels)
    other = group_offsets(question, outer, labels)
    cands = []
    for mt in CAP_RE.finditer(question):
        a, b = mt.span()
        words = question[a:b].split()
        while words and words[0].lower() in QUESTION_WORDS:
            a += len(words[0]) + 1
            words = words[1:]
        if not words or b - a < 2:
            continue
        if any(a < y and x < b for x, y in other):
            continue
        cands.append((a, b))
    cands.sort(key=lambda s: (s[1] - s[0], s[0]), reverse=True)
    if own:
        # the label first; the others only matter if its site turns out weak
        return [own] + [[c] for c in cands], "label"
    if cands:
        return [[c] for c in cands], "capital"

    # No capitalised name at all ("the public speaker who spoke about energy
    # development"): fall back to the noun phrases, last first.
    doc = nlp(question)
    chunks = [
        c for c in doc.noun_chunks
        if c.start > 0 and c.root.pos_ != "PRON"
        and not any(t.tag_.startswith("W") for t in c)
    ]
    out = []
    for c in reversed(chunks):
        s = c.start + (1 if doc[c.start].lower_ in DETS else 0)
        out.append([(doc[s].idx, c.end_char)])
    return out, ("chunk" if out else "none")


# ============================================================
# NL: NEGATING THE LAST PHRASE / CLAUSE
# ============================================================
#
# The negation goes on whatever comes LAST in the question, and always as an
# explicit "not" / "n't" (or "without").
#
# 1. If a predicate follows the inner phrase, the question ends with the OUTER
#    step and that verb is negated:
#
#        Where did the artist nominated for X go to school
#     -> Where did the artist nominated for X not go to school
#
# 2. Otherwise the question ends with the INNER phrase, and the link nearest to
#    the topic entity is negated. Scanning left from the entity, the first of
#    these decides:
#
#      naming participle  the concert tour called X     -> not called X
#      relative clause    the country where X is spoken -> isn't spoken
#                         the person who attended X     -> didn't attend
#      participle         the actress nominated for X   -> not nominated for X
#      "with"             the team with mascot X        -> without mascot X
#
#    Only when the scan reaches the main clause without meeting one of those is
#    the noun phrase negated as a whole:
#
#      place PP           a person from X               -> a person not from X
#      PP on a noun       the author of X               -> an author that is
#                                                          not the author of X
#      premodifier        the Hocus Pocus author        -> an author that is not
#                                                          the Hocus Pocus author
#      apposition         the fictional universe X      -> a fictional universe
#                                                          that is not X

REL_WORDS = {"that", "who", "which", "where", "whose", "whom", "when"}
NAMING = {"called", "named", "titled", "entitled", "nicknamed", "dubbed"}
QUOTE_VERBS = {"say", "quote", "state"}
LOC_PREPS = {"from", "in", "at", "on", "near", "inside", "within", "outside",
             "under", "into", "over", "during"}
DETS = {"the", "a", "an", "this", "that", "its", "his", "her", "their", "these", "those"}
NOMINAL_POS = {"DET", "ADJ", "NOUN", "PROPN", "NUM", "PART"}
FINITE_TAGS = {"VBZ", "VBD", "VBP", "MD"}
PLURAL_TAGS = {"NNS", "NNPS"}
DO_FORMS = {"do", "does", "did"}

# Title words that may sit inside a name without being capitalised.
JOINERS = {"of", "the", "a", "an", "s", "t", "and", "for", "de", "la", "le", "del",
           "du", "von", "van", "da", "vs", "in", "on", "at", "to", "with", "from",
           "el", "al", "y", "e", "di", "der", "den", "des", "dos", "das", "im", "am"}

# Adverb-like words that sit between an auxiliary and its verb ("did the team
# last win", "who once lived"); the negation goes in front of them.
PRE_VERB = {"last", "first", "also", "once", "ever", "currently", "still", "most",
            "recently", "originally", "previously", "formerly", "later", "finally"}

# Negation sites that come from a clear modifier of the ?c noun. The rest are
# kept but marked low-confidence.
STRONG_SITES = {"naming", "relative", "relative_verb", "participle", "with", "pp_not",
                "place_prep", "apposition", "copula_predicate", "premodifier",
                "entity_subject_copula"}

NEGATION_RE = re.compile(
    r"\b(?:not|no|never|neither|nor|none|without|cannot)\b|n't\b", re.IGNORECASE)


def _joined_capital(doc, k, step):
    """Index of a capitalised token reached from k through up to three joiners."""
    stop = QUESTION_WORDS | REL_WORDS
    for _ in range(3):
        if not (0 < k < len(doc)) or doc[k].lower_ not in JOINERS:
            return None
        k += step
        if 0 < k < len(doc) and doc[k].text[:1].isupper() and doc[k].lower_ not in stop:
            return k
    return None


def extend_entity(doc, first, last):
    """
    Grow the entity over the rest of the title it sits in.

    Labels are induced from the dataset and are often only part of a title:
    "Come Easy" for "It Don t Come Easy", "Artist of a Generation" for "Brit
    Award for Artist of a Generation", "Heaven" for "Heaven on a Sunday".
    Scanning from the fragment would find the title's own words as the trigger.
    A quotation ("the person who said Punishment is justice for the unjust")
    runs to the end of the question, so its verbs are not mistaken for a
    trailing clause.
    """
    i = first.i
    while i - 1 > 0:
        if doc[i - 1].text[:1].isupper():
            i -= 1                     # "How to Die in Oregon", "The Rose That Grew"
            continue
        k = _joined_capital(doc, i - 1, -1)
        if k is None:
            break
        i = k
    j = last.i
    while j + 1 < len(doc):
        if doc[j + 1].text[:1].isupper():
            j += 1
            continue
        k = _joined_capital(doc, j + 1, +1)
        if k is None:
            break
        j = k
    # a quotation ("who said ...", "quoted the saying ...") runs to the end
    for t in reversed(doc[:i]):
        if is_rel(doc, t) or t.pos_ == "AUX" and t.i < i - 6:
            break
        if t.lemma_.lower() in QUOTE_VERBS or t.lower_ in {"quote", "saying", "slogan"}:
            j = len(doc) - 1
            break
    return doc[i], doc[j]


def is_rel(doc, t):
    """Whether t opens a relative clause, as opposed to being the question's wh-word."""
    low = t.lower_
    if low not in REL_WORDS or t.i == 0 or t.i + 1 >= len(doc):
        return False
    prev, nxt = doc[t.i - 1], doc[t.i + 1]
    if low == "that":
        # "the team that won" but not "that city"
        return ((prev.pos_ in {"NOUN", "PROPN"} or prev.text[:1].isupper())
                and nxt.pos_ not in {"NOUN", "ADJ"})
    if low == "which":
        if prev.pos_ in {"NOUN", "PROPN"}:
            return True
        # "the country in which X is made", not "In which country ..."
        return (prev.pos_ == "ADP" and t.i >= 2
                and not (nxt.pos_ in {"NOUN", "ADJ"} and nxt.text.islower()))
    # who / whom / whose / where / when past the opening words
    return t.i >= 2


def clause_kind(doc, v):
    """'rel' when the finite verb v belongs to a relative clause, else 'main'."""
    j = v.i - 1
    while (j >= 0 and (doc[j].pos_ in {"ADV", "ADP"} | NOMINAL_POS or doc[j].lower_ in DETS)
           and not is_rel(doc, doc[j])):
        j -= 1                         # "who [at one point] resided"
    return "rel" if j >= 0 and is_rel(doc, doc[j]) else "main"


def is_postnominal_participle(doc, t):
    """ "the actress [nominated] for ...", not "an [appointed] position" """
    if t.pos_ not in {"VERB", "AUX"} or t.i == 0:
        return False
    j = t.i - 1
    while j > 0 and doc[j].pos_ == "ADV":          # "the team currently coached by"
        j -= 1
    prev = doc[j]
    after_noun = prev.pos_ in {"NOUN", "PROPN", "NUM"} or prev.text[:1].isupper()
    if t.tag_ == "VBD" and t.pos_ == "VERB" and after_noun and clause_kind(doc, t) == "main" and (
            t.dep_ == "acl"
            or any(x.pos_ == "AUX" for x in doc[:t.i - 1])):
        # spaCy reads "the actor nominated" as past tense; with the clause's
        # auxiliary already spent ("Where did the artist nominated for X go"),
        # it can only be a participle.
        return True
    if t.tag_ not in {"VBN", "VBG"}:
        return False
    # Only an auxiliary right before the verb makes it finite ("was
    # nominated"); in "Where did the artist nominated for X go" spaCy hangs the
    # main clause's "did" on it too.
    if adjacent_aux(doc, t) is not None:
        return False
    nxt = doc[t.i + 1] if t.i + 1 < len(doc) else None
    if nxt is not None and nxt.pos_ in {"NOUN", "ADJ"} and t.dep_ == "amod":
        return False
    return after_noun


def adjacent_aux(doc, t):
    """The auxiliary directly in front of verb t ("was [nominated]"), if any."""
    for c in t.children:
        if (c.dep_ in {"aux", "auxpass"} and c.i < t.i
                and all(x.pos_ in {"ADV", "PART"} for x in doc[c.i + 1:t.i])):
            return c
    return None


def subject_is_plural(doc, v):
    subj = next((c for c in v.children if c.dep_ in {"nsubj", "nsubjpass"}), None)
    if subj is None:
        subj = next((t for t in reversed(doc[:v.i]) if t.pos_ in {"NOUN", "PROPN"}), None)
    if subj is None:
        return False
    return subj.tag_ in PLURAL_TAGS or subj.lower_ in {"they", "we", "you", "people"}


def negate_verb(doc, v):
    """
    Edit (start, end, replacement) negating the finite verb or auxiliary v.

    Possessive "has" takes do-support ("that doesn't have Puntland", not
    "hasn't Puntland"). An inverted do-form inside a relative clause is put
    back in order: "who did Kat Graham play" -> "who Kat Graham didn't play".
    Adverbs in front of a do-supported verb follow the auxiliary: "that
    slightly contains Italy" -> "that doesn't slightly contain Italy".
    """
    low, lemma = v.lower_, v.lemma_.lower()
    end = v.idx + len(v.text)

    if low in {"has", "have", "had"}:
        nxt = next((t for t in doc[v.i + 1:] if t.pos_ != "ADV"), None)
        if nxt is None or nxt.tag_ != "VBN":
            rep = {"has": "doesn't have", "have": "don't have", "had": "didn't have"}[low]
            return v.idx, end, rep

    if low in DO_FORMS:
        rep = CONTRACTIONS[low]
        # The verb this do-form supports: the next lowercase verb after it.
        h = next((t for t in doc[v.i + 1:]
                  if t.pos_ in {"VERB", "AUX"} and t.text.islower()), None)
        # If the main clause opened with its own do-form ("Where did the artist
        # who did The Audacity of Hope go to school"), that verb is the main
        # clause's, and this "did" is the lexical verb "do".
        main_do = any(t.lower_ in DO_FORMS for t in doc[:v.i])
        if h is None or main_do:
            return v.idx, end, rep + " do"
        if h.i > v.i + 1:
            subj = doc[v.i + 1:h.i]
            if all(t.pos_ in NOMINAL_POS | {"PRON"} or t.text[:1].isupper() for t in subj):
                s_end = subj[-1].idx + len(subj[-1].text)
                return v.idx, s_end, doc.text[subj[0].idx:s_end] + " " + rep
        return v.idx, end, rep

    if low in CONTRACTIONS and (v.pos_ == "AUX" or lemma == "be" or v.tag_ == "MD"):
        return v.idx, end, CONTRACTIONS[low]

    if v.tag_ == "VBD":
        aux = "didn't"
    elif v.tag_ == "VBZ":
        aux = "doesn't"
    elif v.tag_ in {"VBP", "VB"}:
        # agree with the subject: "the artist who record X" -> "doesn't"
        aux = "don't" if subject_is_plural(doc, v) else "doesn't"
    elif v.tag_ in {"VBN", "VBG"}:
        return v.idx, v.idx, "not "
    else:
        return None

    j = v.i - 1
    while (j >= 0 and doc[j].text.islower()
           and (doc[j].pos_ == "ADV" or doc[j].lower_ in PRE_VERB)
           and doc[j].lower_ not in REL_WORDS | {"not", "never"}):
        j -= 1
    if j < v.i - 1:
        adverbs = doc.text[doc[j + 1].idx:v.idx]
        return doc[j + 1].idx, end, aux + " " + adverbs + lemma
    return v.idx, end, aux + " " + lemma


def main_aux_before(doc, subject_start):
    """
    The main clause's auxiliary when it sits in front of the subject ("Where
    [did] the artist ... go", "In what time zone [is] the newspaper X
    distributed"). An auxiliary after a relative word belongs to that relative
    clause ("The nation that [has] Pernambuco ... uses") and does not count.
    """
    for t in doc[:subject_start]:
        if is_rel(doc, t):
            return None
        if t.lower_ in DO_FORMS or (t.pos_ == "AUX" and t.text.islower()):
            return t
    return None


def negate_predicate(doc, h, subject_start):
    """
    Edit negating the predicate headed by verb h, where the subject begins at
    token index subject_start.

    An inverted question keeps its auxiliary up front and takes "not" before
    the verb ("Where did the artist ... not go to school", "Where is the state
    ... not located"); a plain finite verb takes do-support ("The person who
    attended X didn't play for what team").
    """
    lead = main_aux_before(doc, subject_start)
    # After a do-form the verb is always bare ("does X have", "did X get
    # shot"), whatever spaCy tags it. After be/have it may be a finite verb of
    # its own ("What was the party that the person ... was affiliated with").
    if lead is not None and (lead.lower_ in DO_FORMS or not (
            h.pos_ == "AUX" and adjacent_aux(doc, h) is None and h.tag_ in FINITE_TAGS)):
        j = h.i
        while j - 1 >= 0 and doc[j - 1].text.islower() and (
                doc[j - 1].pos_ == "ADV" or doc[j - 1].lower_ in PRE_VERB):
            j -= 1                                 # "not well known", "not last win"
        start = doc[j].idx
        if lead.lower_ in DO_FORMS and h.tag_ in {"VBD", "VBZ"}:
            # "What years did the team ... won the World Series"
            return start, h.idx + len(h.text), "not " + doc.text[start:h.idx] + h.lemma_
        return start, start, "not "

    if h.pos_ == "AUX":
        hv = h.head
        if hv is not h and hv.pos_ == "VERB" and hv.i > h.i + 1:
            return hv.idx, hv.idx, "not "          # "should I not learn"
        return negate_verb(doc, h)

    aux = adjacent_aux(doc, h)
    if aux is not None:
        return negate_verb(doc, aux)               # "were elected" -> "weren't elected"
    if any(c.dep_ == "aux" and c.lower_ in DO_FORMS and c.i < h.i for c in h.children):
        return h.idx, h.idx, "not "                # "who did James Franco play"
    return negate_verb(doc, h)


def np_negation(doc, left_i, head_i):
    """
    Edit inserting "a <noun> that is not " in front of the noun phrase that
    starts at left_i and is headed by head_i, so "the author of X" becomes
    "an author that is not the author of X".
    """
    toks = list(doc[left_i:head_i + 1])
    k = 0
    while k < len(toks) and (toks[k].lower_ in DETS or toks[k].tag_ == "PRP$"):
        k += 1
    core = toks[k:]
    if not core:
        return None
    text = doc.text[core[0].idx:core[-1].idx + len(core[-1].text)]
    pos = doc[left_i].idx
    return pos, pos, noun_not_prefix(text, core[-1].tag_ in PLURAL_TAGS)


def noun_not_prefix(text, plural):
    if plural:
        return text + " that are not "
    return ("an " if text[:1].lower() in "aeiou" else "a ") + text + " that is not "


def answer_nouns(doc):
    """Nouns naming what the question asks for: "What [types]", "What are the [types]"."""
    out = set()
    for t in doc:
        if t.lower_ in {"what", "which"} and t.head.pos_ in {"NOUN", "PROPN"} and t.head.i > t.i:
            out.add(t.head)
    if len(doc) > 2 and doc[0].lower_ in {"what", "which", "who"} and is_copula(doc[1]):
        for t in doc[2:]:
            if t.pos_ in {"NOUN", "PROPN"}:
                out.add(t)
                break
            if t.pos_ not in {"DET", "ADJ", "NUM"}:
                break
    return out


def is_copula(t):
    return t.lemma_.lower() == "be"


def site(edit, name, first, last, inner_end=None, mode="R", group="INNER"):
    """
    A negation site. mode "R" negates the relation to the entity; mode "E"
    keeps the relation and negates the entity's name ("not called X"). group
    is the SPARQL step the negated words express.
    """
    return {"edit": edit, "site": name, "first": first, "last": last,
            "inner_end": last.i if inner_end is None else inner_end, "mode": mode,
            "group": group}


def negate_inner(question, doc, spans):
    """
    Negation site for the inner phrase whose entity is named at `spans`.

    Returns (site dict or None, reason).
    """
    a = min(s for s, _ in spans)
    b = max(e for s, e in spans if s == a)
    ent = [t for t in doc if a <= t.idx < b]
    if not ent:
        return None, "no_entity_tokens"
    first, last = extend_entity(doc, ent[0], ent[-1])
    ent_ids = set(range(first.i, last.i + 1))

    pp = None          # nearest noun-level preposition: ("place"|"pp", prep, noun)
    appo = None        # noun the entity is in apposition to
    i = first.i - 1
    while i >= 0:
        t = doc[i]
        low = t.lower_

        if is_rel(doc, t):
            for v in doc[t.i + 1:]:
                if v.i in ent_ids:
                    continue
                if v.pos_ in {"VERB", "AUX"} and (v.tag_ in FINITE_TAGS or v.dep_ in {"aux", "auxpass"}):
                    # the clause's own verb group ("is spoken") is part of the
                    # inner phrase, so it is not mistaken for a trailing clause
                    end = v.i
                    if v.head.i > v.i and v.head.pos_ in {"VERB", "AUX"}:
                        end = v.head.i
                    mode = "E" if doc[end].lower_ in NAMING else "R"
                    return site(negate_verb(doc, v), "relative", first, last,
                                max(end, last.i), mode), ""
            return None, "relative_without_verb"

        if low in {"with", "without"} and (
                t.head.pos_ not in {"VERB", "AUX"}
                or (i > 0 and doc[i - 1].pos_ in {"NOUN", "PROPN"})):
            rep = "without" if low == "with" else "with"
            return site((t.idx, t.idx + len(t.text), rep), "with", first, last), ""

        if t.pos_ == "ADP" and i > 0:
            # A noun right before the preposition makes it a noun-level link
            # ("the lyracist for X"), whatever spaCy attached it to. A
            # relative verb or participle further left still wins.
            n = doc[i - 1]
            if (pp is None and n.pos_ == "NOUN" and n.text.islower()
                    and n.tag_ not in {"VBN", "VBG", "JJ", "JJS", "JJR"}):
                pp = ("place" if low in LOC_PREPS else "pp", t, n)
            i -= 1
            continue

        if t.pos_ in {"VERB", "AUX"}:
            if low in NAMING or (low == "known" and i + 1 < len(doc) and doc[i + 1].lower_ == "as"):
                # "the concert tour called X" -> "not called X"; "where the
                # government is called X" -> "isn't called X"
                aux = adjacent_aux(doc, t)
                edit = negate_verb(doc, aux) if aux is not None else (t.idx, t.idx, "not ")
                return site(edit, "naming", first, last, mode="E"), ""
            if is_postnominal_participle(doc, t):
                return site((t.idx, t.idx, "not "), "participle", first, last), ""
            if t.dep_ == "amod":
                i -= 1
                continue
            aux = adjacent_aux(doc, t)
            if t.tag_ in {"VBN", "VBG"} and aux is not None and clause_kind(doc, aux) == "rel":
                return site(negate_verb(doc, aux), "relative_verb", first, last), ""
            if clause_kind(doc, t) == "rel":
                return site(negate_verb(doc, t), "relative_verb", first, last), ""
            break                                   # reached the main clause

        if low in {"what", "which", "who", "whom", "where", "when", "how"} and (
                t.i == 0 or t.text.islower()):
            break
        if appo is None and pp is None and t.pos_ in {"NOUN", "PROPN"} and t.i == first.i - 1:
            appo = t                                # "the fictional universe Shameless"
        i -= 1

    # A preposition on the answer noun links ?x to ?c -- the outer step: in
    # "What are the government types in the setting X" negating "in" would
    # negate the wrong half.
    if pp is not None and pp[2] in answer_nouns(doc):
        pp = None

    # The entity premodifies a common noun: "the Hocus Pocus author", "the
    # Samudaya Sacca religion", "Cass beer".
    k = last.i + 1
    if k < len(doc) and doc[k].lower_ in {"s", "'s"}:
        k += 1
    core = []
    while (k < len(doc) and doc[k].tag_ in {"NN", "NNS"} and doc[k].text.islower()
           and len(doc[k].text) > 2):
        core.append(doc[k])
        k += 1
    # "What college degree did the subject of The Boy King earn": with a
    # do-form waiting for its verb and none left, "earn" is that verb.
    if core and any(t.lower_ in DO_FORMS for t in doc[:first.i]) and not any(
            t.pos_ in {"VERB", "AUX"} for t in doc[k:]):
        core = []
    before = first.i - 1
    if before >= 0 and doc[before].lower_ in DETS:
        before -= 1
    if core and before >= 0 and doc[before].pos_ == "NOUN" and doc[before].text.islower():
        core = []                      # "the movie My Daughter s Secret ..."
    if core and (k == len(doc) or doc[k].pos_ in {"VERB", "AUX", "ADP", "PART", "PUNCT", "NUM"}
                 or doc[k].lower_ in REL_WORDS):
        left = first.i
        if left > 0 and doc[left - 1].lower_ in DETS:
            left -= 1
        text = question[core[0].idx:core[-1].idx + len(core[-1].text)]
        pos = doc[left].idx
        prefix = noun_not_prefix(text, core[-1].tag_ in PLURAL_TAGS)
        return site((pos, pos, prefix), "premodifier", first, last, core[-1].i), ""

    if pp is not None:
        kind, prep, n = pp
        if kind == "place":
            return site((prep.idx, prep.idx, "not "), "place_prep", first, last), ""
        left = np_start(doc, n)
        # "... is a part of X": a predicate nominal, so the copula carries it
        if left > 0 and doc[left].lower_ in {"a", "an"} and is_copula(doc[left - 1]):
            return site(negate_verb(doc, doc[left - 1]), "copula_predicate", first, last), ""
        edit = np_negation(doc, left, n.i)
        if edit is not None:
            return site(edit, "pp_not", first, last), ""

    if appo is not None:
        r = np_negation(doc, np_start(doc, appo), appo.i)
        if r is not None:
            return site((r[0], first.idx, r[2]), "apposition", first, last), ""

    # The entity is the subject of a copula: "The Yale Herald is the newspaper
    # of ..." -> "isn't". With any other verb the trailing clause is negated
    # instead (see negate_question).
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if first.i == 0 and root is not None and root.i > last.i and is_copula(root):
        return site(negate_verb(doc, root), "entity_subject_copula", first, last, root.i), ""

    # The entity hangs straight off a verb or preposition of the main clause
    # ("spoken in Nova Roma", "elected from New Jersey"): negate that verb.
    gov = next((t for t in reversed(doc[:first.i])
                if t.pos_ in {"VERB", "AUX"} and t.text.islower()), None)
    if gov is not None:
        if gov.pos_ == "AUX" and gov.head.pos_ == "VERB" and gov.head.i < first.i:
            gov = gov.head
        edit = negate_predicate(doc, gov, gov.i)
        if edit is not None:
            # "What currency is not accepted in X": the main verb is the outer
            # step; a verb inside a modifier is still the inner one
            main = gov.dep_ == "ROOT" or (
                gov.dep_ in {"aux", "auxpass"} and gov.head.dep_ == "ROOT")
            return site(edit, "entity_complement", first, last,
                        group="OUTER" if main else "INNER"), ""

    return None, "no_site"


def best_negation(question, doc, candidates):
    """
    Try each candidate entity in turn and keep the first with a strong site;
    failing that, the first that could be negated at all.
    """
    fallback = None
    for spans in candidates:
        found, _reason = negate_inner(question, doc, spans)
        if found is None or found["edit"] is None:
            continue
        if found["site"] in STRONG_SITES:
            return found
        if fallback is None:
            fallback = found
    return fallback


def trailing_predicate(doc, after_i):
    """The first verb after token index after_i, i.e. a clause that follows the inner phrase."""
    for t in doc[after_i + 1:]:
        # whole words only: spaCy splits "im" (as in "Vater unser im
        # Himmelreich") into "i" + "m" and reads the "m" as a verb
        whole = t.idx == 0 or doc.text[t.idx - 1] == " "
        if t.pos_ in {"VERB", "AUX"} and t.text.islower() and len(t.text) > 1 and whole:
            return t
    return None


def stacks_negation(question, edit):
    """Whether the edit lands next to an existing negation ("shall not not fade")."""
    start, end, rep = edit
    around = question[max(0, start - 8):start] + " " + rep + " " + question[end:end + 8]
    return bool(re.search(r"\bnot\s+not\b|n't\s+not\b|\bnot\s+\w+n't\b", around))


def negate_question(question, doc, candidates):
    """
    Negate the last phrase or clause of the question.

    Returns (site dict, group) with group "INNER" or "OUTER", or (None, None).
    """
    inner = best_negation(question, doc, candidates)

    # Where the inner phrase ends. Without a located entity there is nothing
    # to anchor on, and the last verb of the question is the last clause.
    if inner is not None:
        first, after = inner["first"], inner["inner_end"]
    elif candidates:
        a = min(s for s, _ in candidates[0])
        b = max(e for s, e in candidates[0] if s == a)
        ent = [t for t in doc if a <= t.idx < b]
        if not ent:
            return None, None
        first, last = extend_entity(doc, ent[0], ent[-1])
        after = last.i
    else:
        return None, None

    h = trailing_predicate(doc, after)
    lead = main_aux_before(doc, first.i)
    if h is None and lead is not None and lead.lower_ in DO_FORMS:
        # "Which ocean does the Swan Hill river flow into": the do-form still
        # needs its verb, so the last content word is it
        h = next((t for t in reversed(doc[after + 1:])
                  if t.pos_ in {"NOUN", "VERB", "ADJ"} and t.text.islower()), None)
    if h is not None:
        edit = negate_predicate(doc, h, first.i)
        if edit is not None and not stacks_negation(question, edit):
            return site(edit, "trailing_clause", first, doc[after], len(doc) - 1), "OUTER"
    if inner is not None:
        return inner, inner["group"]
    return None, None


def np_start(doc, n):
    """Index of the first token of the noun phrase headed by n."""
    j = n.i
    while (j - 1 >= 0
           and (doc[j - 1].pos_ in NOMINAL_POS or doc[j - 1].lower_ in DETS)
           and doc[j - 1].lower_ not in REL_WORDS | {"what", "which"}):
        j -= 1
        if doc[j].lower_ in DETS or doc[j].pos_ == "DET":
            break                      # "the person [the film] X is about"
    return j


def apply_edit(question, edit):
    start, end, rep = edit
    if start == 0 and question[:1].isupper():
        # keep sentence case: "The Hocus ..." -> "An author that is not the Hocus ..."
        rep = rep[:1].upper() + rep[1:]
        word = question[end:].split(" ", 1)[0]
        if word.lower() in DETS:
            question = question[:end] + word.lower() + question[end + len(word):]
    return question[:start] + rep + question[end:]


# ============================================================
# SPARQL: NEGATING ONE STEP
# ============================================================

def negate_sparql(parsed, negate, keep, binder=None):
    """
    Wrap `negate` in FILTER NOT EXISTS, leaving `keep` and every filter in the
    main pattern.

    `binder` is an extra triple that keeps ?x bound when the step that bound
    it is the one negated: for "Where did the artist ... not go to school" the
    answers are institutions anyone attended, minus the ones this artist did.

    Returns (query, ok). ok is False when nothing left in the main pattern
    would bind ?x.
    """
    if not negate:
        return None, False

    lines = parsed["lines"]
    body_start, body_end = parsed["body"]

    binds_x = any(t["s"] == "?x" or t["o"] == "?x" for t in keep) or any(
        u["kind"] == "type" and u["s"] == "?x" for u in parsed["units"])
    if not binds_x and binder is None:
        return None, False

    negate_ids = {id(t) for t in negate}
    inside_vars = set().union(*[t["vars"] for t in negate])
    kept_vars = set().union(*[t["vars"] for t in keep]) | {"?x", "?c"} if keep else {"?x", "?c"}
    private = inside_vars - kept_vars

    inside, outside = [], []
    for u in parsed["units"]:
        if u["kind"] == "triple":
            (inside if id(u) in negate_ids else outside).append(u)
        elif u["kind"] == "filter":
            # a filter follows the variables it constrains
            (inside if (u["vars"] & private) else outside).append(u)
        else:
            outside.append(u)

    if not inside:
        return None, False

    def emit(units, indent=""):
        out = []
        for u in units:
            for i in u["idx"]:
                out.append(indent + lines[i].strip())
        return out

    out = lines[:body_start]
    out.extend(emit(outside))
    if binder is not None:
        out.append(binder)
    out.append("FILTER NOT EXISTS {")
    out.extend(emit(inside, "  "))
    out.append("}")
    out.extend(lines[body_end:])
    return "\n".join(out), True


def outer_binder(path):
    """A triple ranging ?x over everything the outer step could return."""
    final = next((t for t in path if t["o"] == "?x"), None)
    if final is None:
        return None
    return f"?x_range {final['p']} ?x ."


def negate_entity_sparql(parsed, inner, mid):
    """
    Keep the inner relation but exclude the named entity: for "the concert tour
    not called Demi World Tour",

        ?c ns:music.artist.concert_tours ?neg_e .
        FILTER (?neg_e != ns:m.010pzj9k)
    """
    target = next((t for t in inner if t["o"] == "ns:" + mid), None)
    if target is None:
        return None
    lines = list(parsed["lines"])
    i = target["idx"][0]
    lines[i] = f"{target['s']} {target['p']} ?neg_e ."
    lines.insert(i + 1, f"FILTER (?neg_e != ns:{mid})")
    return "\n".join(lines)


def entity_mid(question, inner, first, last, labels):
    """The MID of the inner entity named between tokens first..last."""
    mids = []
    for t in inner:
        mids += MID_RE.findall(t["o"])
    mids = list(dict.fromkeys(mids))
    if len(mids) == 1:
        return mids[0]
    lo, hi = first.idx, last.idx + len(last.text)
    for mid in mids:
        forms = (labels or {}).get(mid)
        if forms and any(lo <= a < hi for a, _ in locate_spans(question, label_patterns(forms))):
            return mid
    return None


ORDER_RE = re.compile(r"ORDER\s+BY\s+(.*)", re.IGNORECASE | re.DOTALL)


def order_var_unbound(query):
    """True when ORDER BY sorts on a variable that now only occurs inside FILTER NOT EXISTS."""
    m = ORDER_RE.search(query)
    if not m:
        return False
    main = query[:query.find("FILTER NOT EXISTS")] if "FILTER NOT EXISTS" in query else query
    return any(v not in main for v in VAR_RE.findall(m.group(1)))


# ============================================================
# OUTPUT SANITY CHECKS
# ============================================================

# An unnegated do-form auxiliary still governing the clause that was negated,
# e.g. "Where does mascot Crazy Crab s sports team doesn't play their home
# games". A relative pronoun in between means they are separate clauses and the
# sentence is fine, so those are excluded.
DOUBLED_AUX_RE = re.compile(
    r"\b(?:did|does|do)\s+"
    r"(?:(?!\b(?:that|who|which|where|whose|whom|when|and|but)\b)\w+\s+){0,12}?"
    r"(?:does|do|did)n't\b",
    re.IGNORECASE,
)

GRAMMAR_CHECKS = [
    ("doubled_auxiliary", DOUBLED_AUX_RE),
    # Only do-forms: "wasn't the President of ..." and "isn't The Internationale"
    # are ordinary copulas with a capitalised complement, not stranded subjects.
    ("negation_before_subject",
     re.compile(r"\b(?:did|does|do)n't\s+(?:the\s+|a\s+)?[A-Z]")),
    ("infinitive_contraction", re.compile(r"\bto\s+\w+n't\b", re.IGNORECASE)),
    ("dangling_negation",
     re.compile(r"(?<!\bis)(?<!\bare)(?<!\bwas)(?<!\bwere)n't\s*$", re.IGNORECASE)),
    ("double_negative", re.compile(r"\bnot\s+not\b|\bnot\s+\w+n't\b", re.IGNORECASE)),
]


def grammar_warning(question):
    """
    Name the first thing that makes a negated question ungrammatical.

    These come from parse failures on text that has no punctuation and no
    reliable capitalisation, and a wrong negation is worse than none: the row
    is flagged for review rather than presented as clean.
    """
    for name, pattern in GRAMMAR_CHECKS:
        if pattern.search(question):
            return name
    return ""


# ============================================================
# PER-ROW PIPELINE
# ============================================================

def process_row(question, sparql, labels=None):
    question = str(question)
    sparql = str(sparql)

    result = {
        "negated_question": question,
        "negated_sparql": sparql,
        "negated_clause": "",
        "negated_group": "",
        "group_evidence": "",
        "question_negated": False,
        "sparql_negated": False,
        "negation_status": "",
        "grammar_warning": "",
    }

    # A row whose query cannot be rewritten keeps its question as well, so a
    # pair is never half negated.
    if is_complex(sparql):
        result["negation_status"] = "sparql_shape_unsupported"
        return result
    parsed = parse_body(sparql)
    if parsed is None:
        result["negation_status"] = "sparql_unparsed"
        return result
    path, inner, xonly = split_composition(parsed["units"])
    if not path or not inner:
        result["negation_status"] = "not_a_composition"
        return result

    candidates, source = entity_candidates(question, inner, path + xonly, labels)
    doc = nlp(question)
    found, group = negate_question(question, doc, candidates)
    if found is None or found["edit"] is None:
        result["negation_status"] = "question_unresolved"
        return result

    # The SPARQL negates the same step the question does.
    mode = found["mode"]
    if group == "OUTER":
        has_x = bool(xonly) or any(
            u["kind"] == "type" and u["s"] == "?x" for u in parsed["units"])
        binder = None if has_x else outer_binder(path)
        negated_s, ok = negate_sparql(parsed, path, inner + xonly, binder)
    else:
        negated_s, ok = None, False
        if mode == "E":
            mid = entity_mid(question, inner, found["first"], found["last"], labels)
            negated_s = negate_entity_sparql(parsed, inner, mid) if mid else None
            ok = negated_s is not None
        if not ok:
            mode = "R"
            negated_s, ok = negate_sparql(parsed, inner, path + xonly)
    if not ok:
        result["negation_status"] = "sparql_not_negatable"
        return result

    edit = found["edit"]
    negated_q = apply_edit(question, edit)
    if not NEGATION_RE.search(negated_q):
        result["negation_status"] = "no_negation_word"
        return result

    clause_end = len(question) if group == "OUTER" else max(
        edit[1], doc[found["inner_end"]].idx + len(doc[found["inner_end"]].text))

    result["negated_question"] = negated_q
    result["negated_sparql"] = negated_s
    result["negated_clause"] = question[edit[0]:clause_end]
    result["negated_group"] = group + ("_ENTITY" if group == "INNER" and mode == "E" else "")
    result["group_evidence"] = f"{found['site']}/{source}"
    result["question_negated"] = True
    result["sparql_negated"] = True
    confident = found["site"] in STRONG_SITES or (
        found["site"] == "trailing_clause" and source in {"label", "capital"})
    if order_var_unbound(negated_s):
        confident = False
    result["negation_status"] = "ok" if confident else "ok_low_confidence"
    result["grammar_warning"] = grammar_warning(negated_q)
    return result


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Reading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, encoding=INPUT_ENCODING)
    print(f"Rows: {len(df)}")

    for col in ("original_question", "answers", "sparql"):
        if col in df.columns:
            df[col] = df[col].map(fix_mojibake)

    print("Inducing entity labels from the questions...")
    labels = induce_labels(df["original_question"], df["sparql"])
    print(f"  resolved {len(labels)} MIDs")

    records = []
    for idx, row in df.iterrows():
        if idx and idx % 2000 == 0:
            print(f"  ... {idx}")
        try:
            records.append(
                process_row(row["original_question"], row["sparql"], labels))
        except Exception as exc:
            print(f"[ERROR] row={idx} id={row.get('id')}: {exc}", file=sys.stderr)
            records.append({
                "negated_question": str(row["original_question"]),
                "negated_sparql": str(row["sparql"]),
                "negated_clause": "",
                "negated_group": "",
                "group_evidence": "",
                "question_negated": False,
                "sparql_negated": False,
                "negation_status": f"error: {exc}",
                "grammar_warning": "",
            })

    out = pd.DataFrame(records, index=df.index)
    for col in out.columns:
        df[col] = out[col]

    df.to_csv(OUTPUT_FILE, index=False, encoding=OUTPUT_ENCODING)

    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Input rows:         {len(df)}")
    print(f"Questions negated:  {int(df['question_negated'].sum())}")
    print(f"SPARQL negated:     {int(df['sparql_negated'].sum())}")
    mismatched = int((df["question_negated"] != df["sparql_negated"]).sum())
    print(f"Half-negated pairs: {mismatched}")
    print()
    print("By status:")
    for status, n in df["negation_status"].value_counts().items():
        print(f"  {status:32s} {n}")
    print()
    print("Negation site (site/entity source):")
    for e, n in df["group_evidence"].value_counts().items():
        print(f"  {e:32s} {n}")

    print()
    print("Grammar warnings:")
    warned = df[df["grammar_warning"].astype(str) != ""]
    for w, n in df["grammar_warning"].value_counts().items():
        if w:
            print(f"  {w:32s} {n}")
    print(f"  {'(clean)':32s} {len(df) - len(warned)}")

    review = df[
        (df["negation_status"] != "ok")
        | (df["grammar_warning"].astype(str) != "")
    ]
    if len(review):
        review.to_csv(REVIEW_FILE, index=False, encoding=OUTPUT_ENCODING)
        print(f"\n{len(review)} row(s) need review -> {REVIEW_FILE}")

    print("\nOutput:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
