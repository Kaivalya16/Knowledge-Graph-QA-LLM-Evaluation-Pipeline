#!/usr/bin/env python3
"""
Build negated (question, SPARQL) pairs for the CWQ *conjunction* subset.

Each question in this subset applies two constraints to the answer variable:

    Which school [with the fight song The Orange and Blue] [did Emmitt Smith play for]
                  \_____________ constraint A ___________/ \_____ constraint B ____/

In SPARQL those appear as two groups of triples:

    G1  a chain from a concrete topic entity down into ?x
            ns:m.01480y ns:sports.pro_athlete.teams ?y .
            ?y ns:sports.sports_team_roster.team ?x .

    G2  extra constraints hanging off ?x
            ?x ns:sports.sports_team.fight_song ns:m.04f__h7 .

RULE (word order): negate the LAST clause of the question -- the one after the
boundary that separates it ("did", "does", "that", "and", ...) -- and negate the
matching SPARQL group, wrapping it in FILTER NOT EXISTS while the other group
stays in the main pattern so that ?x remains bound.

Deciding *which* group the last clause expresses needs the MID -> name mapping
that Freebase would provide and that is not available here, so it is inferred
from anchors that do surface in the question:

  * a literal object of G2 ("calling code 508", "3410 post graduates")
  * words from a G2 predicate ("capital", "fight_song", "number_of_episodes")
  * words from a G1 predicate (rarer -- G1 predicates such as
    music.artist.origin are usually realised as unrelated wording, e.g.
    "was the birthplace of Rihanna")

An anchor inside the negated clause identifies that clause's group; an anchor
outside it identifies the other one. Rows where no anchor decides are written to
needs_review.csv with the question negated but the SPARQL left alone, rather
than being given a guessed -- and possibly wrong -- query.
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

INPUT_FILE = os.path.join(BASE_DIR, "qa_dataset_conjunction.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "qa_dataset_conjunction_fixed.csv")
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

# Constructs the line-based reader cannot safely rearrange.
COMPLEX_RE = re.compile(r"\bUNION\b|\bOPTIONAL\b|\bORDER\s+BY\b", re.IGNORECASE)


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

        m = TRIPLE_RE.match(text) if len(idx) == 1 else None
        if m:
            subj, pred, obj = m.group(1), m.group(2), m.group(3).strip()
            kind = "type" if pred in TYPE_PREDICATES else "triple"
            units.append({
                "idx": idx, "kind": kind, "s": subj, "p": pred, "o": obj,
                "vars": set(VAR_RE.findall(text)),
            })
            continue

        units.append({"idx": idx, "kind": "other", "vars": set(VAR_RE.findall(text))})

    return {"lines": lines, "body": (body_start, body_end), "units": units}


def split_groups(units):
    """
    Split the triple units into G1 (chain from a concrete entity into ?x) and
    G2 (the rest, i.e. constraints hanging off ?x).

    The walk deliberately stops at ?x: expanding through it would pull the
    ?x-rooted constraints into G1 and collapse the two clauses into one.
    """
    triples = [u for u in units if u["kind"] == "triple"]
    if not triples:
        return [], []

    g1, used = [], set()
    frontier = {t["s"] for t in triples if t["s"].startswith("ns:")}
    while frontier:
        nxt = set()
        for t in triples:
            if id(t) in used:
                continue
            if t["s"] in frontier:
                g1.append(t)
                used.add(id(t))
                if t["o"] != "?x":
                    nxt.add(t["o"])
        frontier = nxt

    g2 = [t for t in triples if id(t) not in used]
    return g1, g2


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


def induce_labels(questions, sparqls, min_support=3, min_coverage=0.5):
    """
    Work out what each Freebase MID is called, using the questions themselves.

    Freebase is gone and this project has no MID -> name file, but a MID that
    appears in several questions is almost always named in each of them: every
    query mentioning ns:m.0345h sits next to a question containing "Germany".
    So for each MID the capitalised phrases of its questions are pooled and the
    one present in most of them wins.

    Labels are what makes the clause -> triple mapping exact rather than
    inferred, so this is the single biggest lever on SPARQL coverage.

    Returns {mid: label} for MIDs seen at least `min_support` times where some
    phrase covers at least `min_coverage` of them.
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
        for q in qs:
            for phrase in {m.group().strip() for m in CAP_RE.finditer(q)}:
                if len(phrase) < 3 or phrase.lower() in QUESTION_WORDS:
                    continue
                counts[phrase] = counts.get(phrase, 0) + 1

        best = None
        for phrase, n in counts.items():
            coverage = n / len(qs)
            if coverage < min_coverage:
                continue
            # prefer wide coverage, then the more specific (longer) phrase
            key = (round(coverage, 2), len(phrase.split()), len(phrase))
            if best is None or key > best[0]:
                best = (key, phrase, coverage)
        if best:
            scored[mid] = (best[1], best[2])

    # A phrase that won for two different MIDs identifies neither of them.
    owners = {}
    for mid, (phrase, coverage) in scored.items():
        owners.setdefault(phrase.lower(), []).append((coverage, mid))
    labels = {}
    for phrase_l, cands in owners.items():
        if len(cands) > 1:
            continue
        mid = cands[0][1]
        labels[mid] = scored[mid][0]
    return labels


# ============================================================
# ANCHORS: WHERE EACH GROUP SURFACES IN THE QUESTION
# ============================================================

PRED_STOP = {
    "topic", "object", "common", "type", "name", "base", "user", "freebase",
    "value", "notable", "types", "from", "with", "that", "this", "were",
    "have", "been", "does", "about", "other", "more", "than", "into",
}


def _stem(word):
    """Crude suffix strip so "owner_s" also matches "owned"."""
    for suf in ("ings", "ing", "ers", "er", "ors", "or", "ed", "es", "s"):
        if word.endswith(suf) and len(word) - len(suf) >= 4:
            return word[: -len(suf)]
    return word


def predicate_words(pred):
    """
    Content words of the property segment of a Freebase predicate.

    Only the final segment is used: the middle segment names the *domain type*
    ("location.country.administrative_divisions"), and words like "country",
    "film" or "location" match the question's answer-type noun rather than the
    constraint, which inverts the group decision.
    """
    words = set()
    segment = pred.split(".")[-1]
    for word in re.split(r"[_\W]+", segment):
        low = word.lower()
        if len(low) > 3 and low not in PRED_STOP:
            words.add(_stem(low))
    return words


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


def locate(question, patterns):
    """All match offsets for any of the patterns."""
    hits = []
    for pat in patterns:
        try:
            hits += [m.start() for m in re.finditer(pat, question, re.IGNORECASE)]
        except re.error:
            continue
    return hits


def word_offsets(question, words):
    """Offsets where any of the (stemmed) predicate words occurs."""
    hits = []
    for w in words:
        hits += [
            m.start()
            for m in re.finditer(r"\b" + re.escape(w) + r"\w*", question, re.IGNORECASE)
        ]
    return hits


def answer_type_words(doc):
    """
    Words naming the *type* of the answer ("What religion ...", "Which country
    ..."). A predicate whose property name is one of these ("religion",
    "languages_spoken") matches the interrogative frame rather than either
    clause, so it must not be used as an anchor.
    """
    words = set()
    first = next((t for t in doc if not t.is_punct), None)
    if first is None:
        return words
    for t in doc[first.i:first.i + 5]:
        if t.pos_ in {"NOUN", "PROPN", "ADJ"}:
            words.add(_stem(t.lower_))
        elif t.pos_ in {"VERB", "AUX"}:
            break
    return words


# Only real phrasal modifiers count. Including "compound"/"amod" made almost
# any noun-noun pair a modifier ("a run time of 22"), so the rule fired on the
# very constraint that was being negated and inverted the decision.
MOD_DEPS = {"prep", "acl"}


def answer_noun(doc):
    """
    The noun naming what the question asks for ("Which *school* ...").

    The wh-word is not always at the front: "Abkhazia is part of what country
    close to Russia" and "The city town of Lyon is in what country containing
    ..." both ask about a country named mid-sentence, and anchoring on the
    first few tokens picks up "part" or "city" instead.
    """
    for t in doc:
        if t.tag_ in {"WDT", "WP", "WP$"} and t.head.pos_ in {"NOUN", "PROPN"}:
            return t.head
    for t in doc[:4]:
        if t.pos_ in {"NOUN", "PROPN"}:
            return t
    return None


def modifier_spans(doc, head):
    """
    Spans of the nominal modifiers in the question.

    In 61% of these questions only one constraint is a finite clause; the other
    is a phrase hanging off the answer noun ("Which school [with the fight song
    The Orange and Blue] did Emmitt Smith play for", "What movie [starring Zac
    Efron] that has a character named Owen Lars"). Such a phrase is by
    definition not the finite clause, so whichever group its wording belongs to
    is the one that is NOT negated -- and unlike plain span containment this
    still works when the phrase trails the verb.

    Restricted to prepositional and participial phrases on the answer noun:
    widening it to any nominal modifier anywhere made the rule fire on the
    constraint actually being negated.
    """
    noun = answer_noun(doc)
    if noun is None:
        return []

    spans = []
    for t in noun.children:
        if t.dep_ not in MOD_DEPS:
            continue
        sub = list(t.subtree)
        if head in sub:
            continue
        spans.append((min(x.idx for x in sub),
                      max(x.idx + len(x.text) for x in sub)))
    return spans


def prep_spans(doc, head):
    """
    Spans of every prepositional / participial phrase in the question.

    Used only to break a tie when both groups' entities sit in the same clause,
    which happens when the question has a single finite clause and the second
    constraint rides along as a modifier: "Which movies did JJ Abrams direct
    [in Germany]". The entity inside the modifier belongs to the constraint
    that is not being negated.
    """
    spans = []
    for t in doc:
        if t.dep_ not in {"prep", "acl"}:
            continue
        sub = list(t.subtree)
        if head in sub:
            continue
        spans.append((min(x.idx for x in sub),
                      max(x.idx + len(x.text) for x in sub)))
    return spans


def locate_spans(question, patterns):
    """(start, end) of every match of any pattern."""
    spans = []
    for pat in patterns:
        try:
            spans += [m.span() for m in re.finditer(pat, question, re.IGNORECASE)]
        except re.error:
            continue
    return spans


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
                name = labels.get(mid)
                if name:
                    spans += locate_spans(question, [re.escape(name)])
        spans += locate_spans(question, literal_patterns(t["o"]))
    return spans


def decide_group(question, clause_span, g1, g2, type_words=frozenset(), mods=(),
                 labels=None, pmods=()):
    """
    Which group does the negated clause express?

    Returns (group, evidence) with group in {"G1", "G2", None}.
    """
    lo, hi = clause_span

    def inside_outside(offsets):
        ins = any(lo <= o < hi for o in offsets)
        out = any(o < lo or o >= hi for o in offsets)
        return ins, out

    def in_mods(offsets):
        return any(lo2 <= o < hi2 for o in offsets for lo2, hi2 in mods)

    # -1. Entity labels, where they are known. This is the direct evidence:
    #     the group whose entity the clause names is the group it expresses.
    if labels:
        def label_offsets(group):
            offs = []
            for t in group:
                for mid in MID_RE.findall(t["s"]) + MID_RE.findall(t["o"]):
                    name = labels.get(mid)
                    if name:
                        offs += locate(question, [re.escape(name)])
            return offs

        o2, o1 = label_offsets(g2), label_offsets(g1)
        if o2 and o1:
            in2, out2 = inside_outside(o2)
            in1, out1 = inside_outside(o1)
            if in2 and not out2 and out1 and not in1:
                return "G2", "g2_entity_inside"
            if in1 and not out1 and out2 and not in2:
                return "G1", "g1_entity_inside"
            # Both entities in the same clause: the one riding in a modifier
            # belongs to the constraint that is not being negated.
            if pmods:
                def in_p(offs):
                    return any(a <= o < b for o in offs for a, b in pmods)
                p2, p1 = in_p(o2), in_p(o1)
                if p2 and not p1:
                    return "G1", "g2_entity_in_modifier"
                if p1 and not p2:
                    return "G2", "g1_entity_in_modifier"
        if o2 and not o1:
            in2, out2 = inside_outside(o2)
            if in2 and not out2:
                return "G2", "g2_entity_inside"
            if out2 and not in2:
                return "G1", "g2_entity_outside"
        if o1 and not o2:
            in1, out1 = inside_outside(o1)
            if in1 and not out1:
                return "G1", "g1_entity_inside"
            if out1 and not in1:
                return "G2", "g1_entity_outside"

    # 0. If one group's wording sits inside a nominal modifier, that group is
    #    the modifier and the finite clause -- the negated one -- is the other.
    if mods:
        g2m = [o for t in g2 for o in locate(question, literal_patterns(t["o"]))]
        g2_terminal0 = [t for t in g2 if not t["o"].startswith("?")] or g2
        g2m += word_offsets(question, set().union(
            *[predicate_words(t["p"]) for t in g2_terminal0]))
        g1m = [o for t in g1 for o in locate(question, literal_patterns(t["o"]))]
        g1m += word_offsets(question, set().union(
            *[predicate_words(t["p"]) for t in g1])) if g1 else []
        g2_mod, g1_mod = in_mods(g2m), in_mods(g1m)
        if g2_mod and not g1_mod:
            return "G1", "g2_is_modifier"
        if g1_mod and not g2_mod:
            return "G2", "g1_is_modifier"

    # 1. A literal object of G2 is the strongest anchor.
    lit = []
    for t in g2:
        lit += locate(question, literal_patterns(t["o"]))
    if lit:
        ins, out = inside_outside(lit)
        if ins and not out:
            return "G2", "g2_literal_inside"
        if out and not ins:
            return "G1", "g2_literal_outside"

    # 2. Words from the predicates. G2 predicates are attribute-like and
    #    lexicalise well; G1 predicates rarely do, so G2 is checked first.
    # Only the *terminal* triple of a G2 chain names the constraint. For
    #     ?x film.film.starring ?c . ?c film.performance.character ns:m.0dl7n6y
    # that is "character"; including "starring" as well would also match the
    # unrelated "starring Zac Efron" belonging to the other clause.
    g2_terminal = [t for t in g2 if not t["o"].startswith("?")] or g2
    g2w = set().union(*[predicate_words(t["p"]) for t in g2_terminal])
    g1w = set().union(*[predicate_words(t["p"]) for t in g1]) if g1 else set()
    g2_all = set().union(*[predicate_words(t["p"]) for t in g2]) if g2 else set()
    def drop_type_words(words):
        """
        Remove predicate words that merely echo the answer-type noun.

        Stemming is crude, so "countries" -> "countri" and the noun "country"
        do not compare equal; a shared 4-character prefix is close enough.
        """
        out = set()
        for w in words:
            if any(w[:4] == t[:4] for t in type_words):
                continue
            out.add(w)
        return out

    g2w = drop_type_words(g2w - g1w)
    g1w = drop_type_words(g1w - g2_all)

    if g2w:
        ins, out = inside_outside(word_offsets(question, g2w))
        if ins and not out:
            return "G2", "g2_predicate_inside"
        if out and not ins:
            return "G1", "g2_predicate_outside"

    if g1w:
        ins, out = inside_outside(word_offsets(question, g1w))
        if ins and not out:
            return "G1", "g1_predicate_inside"
        if out and not ins:
            return "G2", "g1_predicate_outside"

    # 3. A literal object of G1, as a last resort.
    lit1 = []
    for t in g1:
        lit1 += locate(question, literal_patterns(t["o"]))
    if lit1:
        ins, out = inside_outside(lit1)
        if ins and not out:
            return "G1", "g1_literal_inside"
        if out and not ins:
            return "G2", "g1_literal_outside"

    return None, "no_anchor"


# ============================================================
# NL: FINDING THE LAST CLAUSE
# ============================================================

CLAUSE_DEPS = {"ROOT", "conj", "relcl", "acl", "advcl", "ccomp", "pcomp", "xcomp", "oprd"}
TOP_DEPS = {"ROOT", "conj"}


FINITE_TAGS = {"VBZ", "VBD", "VBP", "MD"}


def find_last_clause(doc):
    """
    Head verb of the last clause, by word order.

    Any finite clause counts, including a trailing relative clause: in
    "Sherrod Brown lives in what city whose county is Lorain" the last clause
    is "whose county is Lorain", and in "Who did Kim Kardashian date that had
    regular tv appearances until 2013" it is the "that had ..." clause. An
    earlier revision skipped those and negated the main verb instead, which
    both read wrong and made the clause span swallow the other constraint.

    Bare participles ("... team named California Golden Bears", "... film shot
    in Chicago") are not clauses for this purpose: they carry no tense of their
    own, and negating them yields text like "not named California Golden Bears
    did Jack London attend". A verb therefore qualifies only when it is finite
    or carries its own auxiliary.
    """
    heads = [
        t for t in doc
        if t.pos_ in {"VERB", "AUX"}
        and t.dep_ in CLAUSE_DEPS
        and (
            t.tag_ in FINITE_TAGS
            or any(c.dep_ in {"aux", "auxpass"} for c in t.children)
        )
    ]
    if not heads:
        # fall back to any clause head at all rather than giving up
        heads = [t for t in doc if t.pos_ in {"VERB", "AUX"} and t.dep_ in CLAUSE_DEPS]
    if not heads:
        return None
    return max(heads, key=lambda t: t.i)


NP_POS = {"DET", "NOUN", "PROPN", "ADJ", "NUM", "PRON", "PART", "ADV", "ADP", "CCONJ"}
DO_FORMS = {"do", "does", "did"}


def clause_aux(doc, head):
    """
    The auxiliary belonging to this clause.

    These questions carry no punctuation or reliable capitalisation, so spaCy
    misparses a fair number of them and the aux dependency alone is not enough:
    in "... did Jack London attend" the "did" comes back as `oprd`. So the
    dependency is tried first and, failing that, the tokens before the verb are
    scanned for a do/be/have form separated from it only by a noun phrase --
    the question-inversion pattern that dominates this dataset.
    """
    auxes = [c for c in head.children if c.dep_ in {"aux", "auxpass"} and c.i < head.i]
    if auxes:
        aux = max(auxes, key=lambda t: t.i)
        for t in doc[aux.i + 1:head.i]:
            if t.pos_ in {"VERB", "AUX"} and t.dep_ in CLAUSE_DEPS:
                return None
        return aux

    # The scan only applies when the verb has no subject of its own. With one,
    # it already heads a complete clause and anything further back belongs to an
    # earlier one: in "Who did Kim Kardashian date that had regular tv
    # appearances until 2013" the last clause is "that had ...", whose subject
    # is "that", and the "did" at the front is a different clause entirely.
    if any(c.dep_ in {"nsubj", "nsubjpass"} for c in head.children):
        return None

    for i in range(head.i - 1, max(head.i - 7, -1), -1):
        t = doc[i]
        if t.lower_ in CONTRACTIONS and t.lemma_.lower() in {"do", "be", "have"}:
            if all(x.pos_ in NP_POS for x in doc[i + 1:head.i]):
                return t
            return None
        if t.pos_ not in NP_POS:
            return None
    return None


def use_aux(head, aux):
    """
    Whether the negation belongs on the auxiliary rather than on the verb.

    A genuinely finite verb does not need one, and trusting a spurious
    attachment turns "whose capital is Baghdad fought in the 1991 Gulf War"
    into "... capital isn't Baghdad fought ...". The exception is do-support:
    "did Jack London attend" gets "attend" tagged as finite by mistake, yet the
    negation still belongs on "did".
    """
    if aux is None:
        return False
    if head.tag_ not in FINITE_TAGS:
        return True
    return aux.lemma_.lower() in {"do"} or aux.lower_ in DO_FORMS


WH_TAGS = {"WDT", "WP", "WP$", "WRB"}


def clause_start(doc, head, aux):
    """
    Where the last clause begins.

    The head verb's subtree is useless here: for a ROOT verb it covers the whole
    sentence, which would put the other clause's wording inside the span and
    bias every decision towards G2. The clause instead starts at whichever comes
    first of:

      * its auxiliary            "... did Emmitt Smith play for"
      * its marker / conjunction "... and has a capital of Montgomery"
      * its relative pronoun     "... that has a character named ..."
      * its subject phrase       "... whose basketball team head coach is ..."
      * the verb itself

    The question's own leading wh-word is skipped: "Where did Dr Seuss ..." is
    the interrogative frame, not a clause boundary, and treating it as one puts
    the whole sentence in the span.
    """
    first_content = next((t.i for t in doc if not t.is_punct), 0)

    starts = [head.idx]
    if aux is not None:
        starts.append(aux.idx)

    for child in head.children:
        if child.i >= head.i:
            continue
        if child.dep_ in {"mark", "cc", "preconj"}:
            starts.append(child.idx)
        elif child.tag_ in WH_TAGS and child.i > first_content:
            starts.append(child.idx)
        elif child.dep_ in {"nsubj", "nsubjpass"}:
            # include the whole subject phrase, but never the leading wh-word
            left = min(t.i for t in child.subtree)
            if left > first_content:
                starts.append(doc[left].idx)

    return min(starts)


def clause_span(doc, head, aux):
    """The last clause runs from its start to the end of the question."""
    return clause_start(doc, head, aux), len(doc.text)


# ============================================================
# NL: NEGATING THE CLAUSE
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

PLURAL_TAGS = {"NNS", "NNPS"}


def _subject_is_plural(doc, head):
    for child in head.children:
        if child.dep_ in {"nsubj", "nsubjpass"}:
            if child.tag_ in PLURAL_TAGS:
                return True
            if child.pos_ in {"NOUN", "PROPN", "PRON"}:
                return child.lower_ in {"they", "we", "you"}
    for chunk in doc.noun_chunks:
        if chunk.root.pos_ in {"NOUN", "PROPN"}:
            return chunk.root.tag_ in PLURAL_TAGS
    return False


def already_negated(doc, head, span):
    lo, hi = span
    for t in doc:
        if lo <= t.idx < hi and (t.lower_ in {"not", "never"} or t.dep_ == "neg"):
            return True
        if lo <= t.idx < hi and t.lower_.endswith("n't"):
            return True
    return False


# Subjects that must not be moved: relative and interrogative pronouns (the
# clause is not inverted at all) plus expletives. Personal pronouns are real
# subjects and do move -- "... do they speak in Ghana" -> "... they don't
# speak in Ghana".
WH_WORDS = {
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "that", "it", "there",
}


def last_content_index(doc):
    """Index of the last non-punctuation token."""
    for t in reversed(doc):
        if not t.is_punct:
            return t.i
    return len(doc) - 1


def inverted_subject(doc, head, pivot):
    """
    Token range of the subject when the question is inverted, else None.

    In "... did Emmitt Smith play for" the auxiliary sits in front of its
    subject, so negating it in place gives "... didn't Emmitt Smith play for",
    which reads as a tag question rather than as a negated constraint. The
    subject has to move in front of the auxiliary first.

    spaCy frequently misattaches the subject in these unpunctuated questions --
    for "Which school ... did Emmitt Smith play for" it makes "school" the
    nsubj and "Smith" a direct object -- so when the tagged subject sits before
    the auxiliary the noun phrase directly after it is used instead.

    A subject that is itself the wh-word is left alone, and so is a predicate
    nominal: "What is the Latin American country with ISO 068" negates to
    "What isn't the Latin American country ...", and moving "the Latin American
    country" in front of "is" would wreck it.
    """
    # With no separate auxiliary, only do-support can be inverted. "that has a
    # character named Owen Lars" and "and has a capital of Montgomery" are
    # ordinary subject-verb order, and "What is the Latin American country ..."
    # is a copula whose complement must stay put.
    #
    # A copula is the exception when the question opens with "where"/"when":
    # there the wh-word is adverbial, so the noun phrase after the verb really
    # is the subject ("Where was the Vietnam War in X" -> "Where the Vietnam
    # War wasn't in X"), unlike "What is ..." where it is the complement.
    if pivot is head:
        lemma = pivot.lemma_.lower()
        lead_wh = next(
            (t for t in doc[:3] if t.tag_ in {"WDT", "WP", "WP$", "WRB"}), None
        )
        adverbial_wh = lead_wh is not None and lead_wh.tag_ == "WRB"
        if lemma != "do" and not (lemma == "be" and adverbial_wh):
            return None
        # A copula whose subject already precedes it is in plain order, so what
        # follows is its complement: "... where the capital is Quito" must not
        # become "... where the capital Quito isn't". Do-support is different --
        # "did JJ Abrams direct" is inverted even though spaCy hangs the answer
        # noun off it as a subject.
        if lemma == "be" and any(
            c.dep_ in {"nsubj", "nsubjpass"} and c.i < pivot.i
            for c in head.children
        ):
            return None

    if pivot is head:
        # A mis-parsed "did <SUBJ> <VERB> ..." -- spaCy has tagged the verb as
        # a noun or adjective, so there is no reliable verb boundary and
        # neither the noun chunk ("Tupac star") nor the entity ("JJ Abrams
        # direct") can be trusted to stop in the right place. Take the run of
        # nominal tokens after the auxiliary, then drop any trailing common
        # nouns once a proper noun has been seen, which is where the misread
        # verb sits.
        def name_like(t):
            # These questions are uncapitalised and unpunctuated, so parts of
            # proper names are often tagged ADJ or NOUN; a capital letter is
            # the more reliable signal ("Kim Kardashian", where "Kardashian"
            # comes back as an adjective).
            return t.pos_ == "PROPN" or (
                t.pos_ in {"ADJ", "NOUN"} and t.text[:1].isupper()
            )

        run = []
        for t in doc[pivot.i + 1:]:
            if t.pos_ not in {"DET", "PROPN", "PRON", "NOUN"} and not name_like(t):
                break
            run.append(t)
        if not run or len(run) > 6:
            return None
        if any(name_like(t) for t in run):
            while run and not name_like(run[-1]):
                run.pop()
        if not run:
            return None
        if any(t.tag_ in {"WDT", "WP", "WP$"} or t.lower_ in WH_WORDS for t in run):
            return None
        if run[-1].i >= last_content_index(doc):
            return None          # nothing would be left after the negation
        return run[0].i, run[-1].i

    subj = [c for c in head.children if c.dep_ in {"nsubj", "nsubjpass"}]
    if subj:
        s = subj[0]
        # A relative or interrogative subject means the clause is not inverted.
        if s.tag_ in {"WDT", "WP", "WP$", "WRB"} or s.lower_ in WH_WORDS:
            return None
        left = min(t.i for t in s.subtree)
        right = max(t.i for t in s.subtree)
        if (
            left > pivot.i
            and not (pivot is not head and right >= head.i)
            and not any(doc[i].pos_ == "VERB" for i in range(left, right + 1))
            and right < last_content_index(doc)
        ):
            return left, right

    # These subjects are nearly always named entities, and the entity boundary
    # is more reliable than the noun chunk: for "... did Tupac star in" spaCy
    # returns the chunk "Tupac star", which would move the verb along with the
    # subject and produce "Tupac star didn't in".
    for ent in doc.ents:
        start = ent.start
        if start == pivot.i + 2 and doc[pivot.i + 1].pos_ == "DET":
            start = pivot.i + 1                    # keep a leading determiner
        elif start != pivot.i + 1:
            continue
        if pivot is not head and ent.end > head.i:
            continue
        if ent.end - 1 >= last_content_index(doc):
            continue
        return start, ent.end - 1

    for chunk in doc.noun_chunks:
        if chunk.start <= pivot.i:
            continue
        if chunk.start > pivot.i + 1:
            break                                  # must follow the pivot directly
        if pivot is not head and chunk.end > head.i:
            return None
        if chunk.root.dep_ in {"attr", "acomp"}:
            return None                            # predicate nominal, not a subject
        if any(t.tag_ in {"WDT", "WP", "WP$"} or t.lower_ in WH_WORDS for t in chunk):
            return None
        if chunk.end - 1 >= last_content_index(doc):
            return None
        return chunk.start, chunk.end - 1

    return None


def uninvert(question, doc, pivot, subject, replacement):
    """
    Rewrite "<pivot> <subject> <rest>" as "<subject> <negated pivot> <rest>".
    """
    left, right = subject
    subj_text = question[doc[left].idx:doc[right].idx + len(doc[right].text)]
    pre = question[:pivot.idx]
    post = question[doc[right].idx + len(doc[right].text):]
    if pre and not pre.endswith(" "):
        pre += " "
    return pre + subj_text + " " + replacement + post


COORD = {"and", "but", "also", "plus"}


def _verb_negation(doc, token, plural_subject):
    """Replacement text for a verb-like token that heads the last constraint."""
    word = token.lower_
    if token.pos_ == "AUX" or token.lemma_.lower() in {"be", "do"}:
        return CONTRACTIONS.get(word, word + " not")
    if token.tag_ in {"VBN", "VBG"}:
        return "not " + token.text            # participle: no do-support
    if token.tag_ == "VBD":
        return "didn't " + token.lemma_
    if token.tag_ == "VBZ" or (token.pos_ == "NOUN" and word.endswith("s")):
        # spaCy reads a conjoined verb as a plural noun ("... and imports good
        # from Japan"), but the lemma is still right.
        return "doesn't " + token.lemma_
    return ("don't " if plural_subject else "doesn't ") + token.lemma_


def _starts_a_predicate(doc, token):
    """
    Whether this token heads a predicate rather than a noun phrase.

    After a coordinating conjunction the conjunct either starts with its verb
    ("and imports good from Japan", "and featued taylor lautner") or is a bare
    noun phrase ("and film job of Vincenzo Proietti"). spaCy mislabels the
    first kind as a noun, so the following token decides it: a verb takes a
    complement, a compound noun is followed by another noun.
    """
    if token.pos_ in {"VERB", "AUX"}:
        return True
    if token.pos_ != "NOUN":
        return False
    nxt = doc[token.i + 1] if token.i + 1 < len(doc) else None
    return nxt is not None and nxt.pos_ not in {"NOUN", "PROPN", "PUNCT"}


def find_negation_site(doc, own, other):
    """
    Where the negation belongs when the last constraint is not a finite clause.

    Two shapes the finite-clause search gets wrong:

      A. a coordination splitting the two constraints -- "What country includes
         the GIlbert Islands and imports good from Japan". The negation goes
         after "and", onto the second conjunct.

      B. a trailing modifier on the answer noun -- "... in what country
         containing Tomtom Western Europe", "... what country close to Russia".
         The negation goes at the front of the modifier ("not containing ...").

    `own` holds the character offsets where the constraint being negated is
    named and `other` those of the constraint kept. Either may be empty --
    usually only one of the two entities has a label -- so a site is accepted
    when it contains every known `own` offset and no `other` offset. Returns
    (token, replacement) or None to fall back to the finite-clause path.
    """
    plural = False
    for chunk in doc.noun_chunks:
        if chunk.root.pos_ in {"NOUN", "PROPN"}:
            plural = chunk.root.tag_ in PLURAL_TAGS
            break

    ent_spans = [(e.start_char, e.end_char) for e in doc.ents]

    # --- A. coordination separating the two constraints -------------------
    for t in reversed(list(doc)):
        if t.lower_ not in COORD or t.dep_ not in {"cc", "advmod", "conj"}:
            continue
        # "The Orange and Blue" is a title, not a coordination of constraints:
        # a real one always follows a complete predicate.
        if not any(x.pos_ in {"VERB", "AUX"} for x in doc[:t.i]):
            continue
        if any(a <= t.idx < b for a, b in ent_spans):
            continue
        # A coordination inside one of the names is part of that name.
        if any(a < t.idx < b for a, b in list(own) + list(other)):
            continue
        nxt = next((x for x in doc[t.i + 1:] if not x.is_punct), None)
        if nxt is None:
            continue
        if any(a >= t.idx for a, _ in other):
            continue                       # kept constraint must sit before it
        if own and not all(a > t.idx for a, _ in own):
            continue                       # negated constraint must sit after
        if not own and not any(x.pos_ in {"VERB", "AUX"} for x in doc[t.i + 1:]):
            # With nothing locatable after the conjunction, only a predicate
            # shows this is a real coordination: "... the book The Fish the
            # Fighters and the Song" is one title, and negating "the Song"
            # would be nonsense.
            continue
        if _starts_a_predicate(doc, nxt):
            return nxt, _verb_negation(doc, nxt, plural)
        return nxt, "not " + nxt.text

    # --- B. trailing modifier on the answer noun --------------------------
    noun = answer_noun(doc)
    if noun is None:
        return None

    best = None
    for t in noun.children:
        if t.i <= noun.i or t.dep_ not in {"acl", "relcl", "advmod", "amod"}:
            continue
        sub = list(t.subtree)
        lo = min(x.idx for x in sub)
        hi = max(x.idx + len(x.text) for x in sub)
        if any(lo <= a < hi for a, _ in other):
            continue                       # would negate the kept constraint too
        if own and not all(lo <= a < hi for a, _ in own):
            continue
        # It has to be the *last* predicate. "What country sharing borders with
        # Spain does the Setubal District belong to" has a leading participle;
        # the constraint to negate is the "belong to" clause after it.
        end_i = max(x.i for x in sub)
        if any(x.pos_ in {"VERB", "AUX"} for x in doc[end_i + 1:]):
            continue
        if best is None or t.i > best.i:
            best = t
    if best is None:
        return None

    if best.pos_ in {"VERB", "AUX"} and best.tag_ not in {"VBN", "VBG"}:
        return best, _verb_negation(doc, best, plural)
    return best, "not " + best.text


def negate_question(question):
    """
    Negate the last clause of the question.

    Returns (negated_question, span, status, type_words, mods, prep_mods).
    """
    doc = nlp(question)
    head = find_last_clause(doc)
    if head is None:
        return question, None, "no_clause_found", frozenset(), (), ()

    aux = clause_aux(doc, head)
    span = clause_span(doc, head, aux)

    if already_negated(doc, head, span):
        return question, span, "already_negated", frozenset(), (), ()

    # 1. An auxiliary carries the negation: "did X play" -> "didn't X play".
    if (
        use_aux(head, aux)
        and aux.lemma_.lower() == "have"
        and head.tag_ == "VBG"
    ):
        # "Which team ... has Tim Howard playing for it": "has" is a main verb
        # with an object and a small clause, not a perfect auxiliary, so it
        # takes do-support and the clause is not inverted ("hasn't playing").
        plural = _subject_is_plural(doc, head)
        target = aux
        replacement = ("don't have" if plural else "doesn't have")
        start = target.idx
        negated = question[:start] + replacement + question[start + len(target.text):]
        return (negated, span, "ok", answer_type_words(doc),
                modifier_spans(doc, head), prep_spans(doc, head))

    # An infinitive cannot carry a contraction: "a fun thing to do in San
    # Diego" negates to "... to not do ...", not "... to don't ...".
    if any(c.tag_ == "TO" or c.lower_ == "to" for c in head.children if c.i < head.i):
        start = head.idx
        negated = question[:start] + "not " + question[start:]
        return (negated, span, "ok", answer_type_words(doc),
                modifier_spans(doc, head), prep_spans(doc, head))

    if use_aux(head, aux) and aux.lower_ in CONTRACTIONS:
        target, replacement = aux, CONTRACTIONS[aux.lower_]
    # 2. The clause head is itself an auxiliary, a copula, or do-support that
    #    spaCy happened to tag as the main verb ("did Tupac star in", where
    #    do-support would otherwise yield the doubled "didn't do").
    elif head.lower_ in CONTRACTIONS and (
        head.pos_ == "AUX"
        or head.lemma_.lower() in {"be", "do"}
    ):
        target, replacement = head, CONTRACTIONS[head.lower_]
    # 3. A finite lexical verb takes do-support.
    elif head.tag_ == "VBD":
        target, replacement = head, "didn't " + head.lemma_
    elif head.tag_ in {"VBZ", "VBP", "VB"}:
        plural = _subject_is_plural(doc, head)
        target = head
        replacement = ("don't " if plural else "doesn't ") + head.lemma_
    # 4. A participle or gerund with no auxiliary in front of it.
    elif head.tag_ in {"VBN", "VBG"}:
        target, replacement = head, "not " + head.text
    else:
        return question, span, "cannot_negate_verb", frozenset(), (), ()

    # When the question is inverted, the subject moves in front of the negated
    # auxiliary instead of the negation being planted before it.
    pivot = aux if target is aux else head
    if target is pivot:
        subject = inverted_subject(doc, head, pivot)
        if subject is not None:
            negated = uninvert(question, doc, pivot, subject, replacement)
            return (negated, span, "ok", answer_type_words(doc),
                    modifier_spans(doc, head), prep_spans(doc, head))

    start = target.idx
    negated = question[:start] + replacement + question[start + len(target.text):]
    return (negated, span, "ok", answer_type_words(doc),
            modifier_spans(doc, head), prep_spans(doc, head))


# ============================================================
# SPARQL: WRAPPING A GROUP IN FILTER NOT EXISTS
# ============================================================

def negate_sparql(parsed, group, other):
    """
    Wrap `group` in FILTER NOT EXISTS, keeping everything else in the main
    pattern.

    A multi-line filter travels with the group whose variables it constrains:
    the roster time qualifiers talk about ?y, so if the ?y chain moves inside
    the NOT EXISTS block they have to move with it or ?y would be unbound.

    Returns (query, ok). ok is False when what remains would not bind ?x, which
    would turn the query into an unbounded negation.
    """
    if not group or not other:
        return None, False

    lines = parsed["lines"]
    body_start, body_end = parsed["body"]

    if not any(t["s"] == "?x" or t["o"] == "?x" for t in other):
        return None, False

    inside_ids = {id(t) for t in group}
    inside_vars = set().union(*[t["vars"] for t in group])
    kept_vars = set().union(*[t["vars"] for t in other]) | {"?x"}
    # variables that exist only within the negated group
    private = inside_vars - kept_vars

    inside, outside = [], []
    for u in parsed["units"]:
        if u["kind"] == "triple":
            (inside if id(u) in inside_ids else outside).append(u)
        elif u["kind"] == "filter":
            # follow the group that owns the variables this filter constrains
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
    out.append("FILTER NOT EXISTS {")
    out.extend(emit(inside, "  "))
    out.append("}")
    out.extend(lines[body_end:])
    return "\n".join(out), True


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
    }

    # The two constraints have to be known before the question can be negated:
    # where each one is named decides which is last, and it is the last one
    # that gets negated.
    g1 = g2 = None
    parsed = None
    if not is_complex(sparql):
        parsed = parse_body(sparql)
        if parsed is not None:
            g1, g2 = split_groups(parsed["units"])

    site = None
    site_group = None
    if g1 and g2:
        o1 = group_offsets(question, g1, labels)
        o2 = group_offsets(question, g2, labels)
        if o1 or o2:
            # Try both readings: usually only one of the two constraints has a
            # locatable name, so the other is identified by exclusion -- the
            # phrase that does *not* contain the known one.
            order = [("G1", o1, o2), ("G2", o2, o1)]
            if o1 and o2:
                order.sort(key=lambda c: max(a for a, _ in c[1]) if c[1] else -1,
                           reverse=True)
            else:
                order.sort(key=lambda c: len(c[1]))   # unknown group first
            doc = nlp(question)
            for last, own, oth in order:
                found = find_negation_site(doc, own, oth)
                if found is not None:
                    site, site_group = found, last
                    break

    if site is not None:
        token, replacement = site
        start = token.idx
        negated_q = question[:start] + replacement + question[start + len(token.text):]
        span = (start, len(question))
        result["negated_question"] = negated_q
        result["question_negated"] = True
        result["negated_clause"] = question[start:]
        group, evidence = site_group, "last_named_constraint"
        type_words, mods, pmods = frozenset(), (), ()
    else:
        negated_q, span, q_status, type_words, mods, pmods = negate_question(question)
        result["negated_question"] = negated_q
        result["question_negated"] = q_status == "ok"
        if span:
            result["negated_clause"] = question[span[0]:span[1]]
        if q_status != "ok":
            result["negation_status"] = q_status
            return result
        group = evidence = None

    if is_complex(sparql):
        result["negation_status"] = "sparql_shape_unsupported"
        return result

    if parsed is None:
        result["negation_status"] = "sparql_unparsed"
        return result

    if not g1 or not g2:
        result["negation_status"] = "single_constraint_group"
        return result

    if group is None:
        group, evidence = decide_group(
            question, span, g1, g2, type_words, mods, labels, pmods)
    result["group_evidence"] = evidence
    if group is None:
        result["negation_status"] = "clause_group_undetermined"
        return result

    result["negated_group"] = group
    chosen, other = (g2, g1) if group == "G2" else (g1, g2)
    negated_s, ok = negate_sparql(parsed, chosen, other)
    if not ok:
        result["negation_status"] = "sparql_would_unbind_x"
        return result

    result["negated_sparql"] = negated_s
    result["sparql_negated"] = True
    result["negation_status"] = "ok"
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
    print()
    print("By status:")
    for status, n in df["negation_status"].value_counts().items():
        print(f"  {status:32s} {n}")
    print()
    print("Negated group:")
    for g, n in df["negated_group"].value_counts().items():
        print(f"  {g or '(none)':32s} {n}")
    print()
    print("Group evidence:")
    for e, n in df["group_evidence"].value_counts().items():
        print(f"  {e:32s} {n}")

    review = df[~df["sparql_negated"].astype(bool) | ~df["question_negated"].astype(bool)]
    if len(review):
        review.to_csv(REVIEW_FILE, index=False, encoding=OUTPUT_ENCODING)
        print(f"\n{len(review)} row(s) need review -> {REVIEW_FILE}")

    print("\nOutput:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
