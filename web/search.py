"""Read-only search over the exported demo bundle (`web/demo_data/demo.sqlite`).
No LLM, no embeddings, no writes: a visitor's input only ever reaches a
parameterised SELECT and a name comparison."""

from __future__ import annotations

import dataclasses
import html
import re
import sqlite3
from pathlib import Path

from c2k.norm import norm
from c2k.query import names_match, parse_query
from c2k.understand import Lexicon, RelationMatcher, Understanding, understand

MAX_INPUT_CHARS = 200


def connect(path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def clean_input(text: str | None) -> str:
    return (text or "").strip()[:MAX_INPUT_CHARS]


def relations(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute("SELECT DISTINCT relation FROM facts ORDER BY relation")]


def meta(conn: sqlite3.Connection) -> dict[str, str]:
    return {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM meta")}


def _entity_matches(entity: str, row: sqlite3.Row) -> bool:
    if not norm(entity):
        return True
    return any(names_match(entity, row[col]) for col in ("subject_canonical", "object_canonical", "subject", "object"))


def search(
    conn: sqlite3.Connection,
    *,
    text: str | None = None,
    entity: str | None = None,
    relation: str | None = None,
    relations: list[str] | None = None,
    source_type: str | None = None,
    limit: int = 200,
) -> list[dict]:
    """Free text (parsed by the same heuristic as `c2k query`) and explicit
    filters combine; an explicit filter wins over what the text implies."""
    text, entity, relation = clean_input(text), clean_input(entity), clean_input(relation)
    if text:
        parsed_relation, parsed_entity = parse_query(text)
        relation = relation or (parsed_relation.value if parsed_relation else "")
        entity = entity or (parsed_entity or "")
        if not (relation or entity or source_type):
            # Text that names neither a relation nor an entity can't be
            # interpreted; returning every fact would look like an answer.
            return []

    sql = "SELECT * FROM facts WHERE 1=1"
    params: list[str] = []
    if relations:
        sql += f" AND relation IN ({','.join('?' * len(relations))})"
        params.extend(relations)
    elif relation:
        sql += " AND relation = ?"
        params.append(relation)
    if source_type:
        sql += " AND source_type = ?"
        params.append(source_type)
    sql += " ORDER BY COALESCE(confidence, 0) DESC, doc_id, fact_id"

    results = []
    for row in conn.execute(sql, params):
        if entity and not _entity_matches(entity, row):
            continue
        results.append(dict(row))
        if len(results) >= limit:
            break
    return results


def group_facts(facts: list[dict]) -> list[dict]:
    """Collapses repeats of the same resolved fact (common: one company's
    headquarters appears in many filings). Keeps the first occurrence and
    lists the other documents it was found in under `also_in`."""
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for f in facts:
        key = (norm(f["subject_canonical"]), f["relation"], norm(f["object_canonical"]))
        groups.setdefault(key, []).append(f)
    out = []
    for members in groups.values():
        first = dict(members[0])
        seen = {first["doc_id"]}
        first["also_in"] = []
        for m in members[1:]:
            if m["doc_id"] not in seen:
                seen.add(m["doc_id"])
                first["also_in"].append({"doc_title": m["doc_title"], "source_url": m["source_url"]})
        out.append(first)
    return out


def lexicon(conn: sqlite3.Connection) -> Lexicon:
    """What the data's entity names look like, used to spot names in a
    question and to leave filler words out of the entity."""
    names = [
        r[0]
        for r in conn.execute(
            "SELECT subject_canonical FROM facts UNION SELECT object_canonical FROM facts "
            "UNION SELECT subject FROM facts UNION SELECT object FROM facts"
        )
    ]
    return Lexicon.from_names(names)


def ask(
    conn: sqlite3.Connection,
    question: str,
    *,
    matcher: RelationMatcher | None = None,
    lex: Lexicon | None = None,
    entity: str | None = None,
    relation: str | None = None,
    source_type: str | None = None,
    limit: int = 200,
) -> tuple[list[dict], Understanding]:
    """Answers a free-text question. The relation(s) and entity come from
    `understand` (keywords plus meaning matching when `matcher` is given); an
    explicit `entity` or `relation` overrides what was understood. Facts are
    ordered so the most likely relation comes first. A question that names no
    relation and no entity gets no results rather than everything."""
    question = clean_input(question)
    u = understand(question, matcher, lex) if question else Understanding((), None, "none")
    relations = [clean_input(relation)] if relation else list(u.relations)
    entity = clean_input(entity) or u.entity
    if not (relations or entity or source_type):
        return [], u
    facts = search(conn, entity=entity, relations=relations or None, source_type=source_type, limit=limit)
    if not facts and relations and entity and not relation:
        # The kind of fact may have been misread (idioms, near-ties). Rather than
        # go silent when the name is known, show everything about it, most
        # likely kind of fact first, and say so.
        facts = search(conn, entity=entity, source_type=source_type, limit=limit)
        if facts:
            u = dataclasses.replace(u, widened=True)
            order = sorted(u.scores, key=lambda r: -u.scores[r])
            relations = list(relations) + [r for r in order if r not in relations]
    rank = {r: i for i, r in enumerate(relations)}
    facts.sort(key=lambda f: rank.get(f["relation"], len(rank)))
    return facts, u


def stats(conn: sqlite3.Connection) -> dict:
    """Headline figures for the page header."""
    one = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
    return {
        "documents": one("SELECT COUNT(DISTINCT doc_id) FROM facts"),
        "facts": one("SELECT COUNT(*) FROM facts"),
        "entities": one(
            "SELECT COUNT(*) FROM (SELECT subject_canonical AS n FROM facts UNION SELECT object_canonical FROM facts)"
        ),
        "fact_types": one("SELECT COUNT(DISTINCT relation) FROM facts"),
        "by_source": dict(conn.execute("SELECT source_type, COUNT(DISTINCT doc_id) FROM facts GROUP BY source_type")),
    }


def facts_by_type(conn: sqlite3.Connection) -> list[tuple[str, int]]:
    """(relation, number of facts), most facts first, ties alphabetical."""
    rows = conn.execute("SELECT relation, COUNT(*) AS n FROM facts GROUP BY relation ORDER BY n DESC, relation")
    return [(r[0], r[1]) for r in rows]


def entity_names(conn: sqlite3.Connection, *, min_facts: int = 2) -> list[str]:
    rows = conn.execute(
        "SELECT name, COUNT(*) AS n FROM ("
        " SELECT subject_canonical AS name FROM facts UNION ALL SELECT object_canonical FROM facts"
        ") GROUP BY name HAVING n >= ? ORDER BY n DESC, name",
        (min_facts,),
    )
    return [r["name"] for r in rows]


def entity_page(conn: sqlite3.Connection, name: str) -> dict:
    """Everything about one resolved entity: the surface strings that were
    merged into it, and its facts grouped by relation."""
    name = clean_input(name)
    rows = [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM facts WHERE subject_canonical = ? OR object_canonical = ? ORDER BY relation, doc_id",
            (name, name),
        )
    ]
    surfaces: set[str] = set()
    for r in rows:
        if r["subject_canonical"] == name:
            surfaces.add(r["subject"])
        if r["object_canonical"] == name:
            surfaces.add(r["object"])
    by_relation: dict[str, list[dict]] = {}
    for r in rows:
        by_relation.setdefault(r["relation"], []).append(r)
    return {"name": name, "surfaces": sorted(surfaces), "by_relation": by_relation, "n_facts": len(rows)}


def html_text(text: str) -> str:
    """Document text made safe to place inside HTML that Streamlit renders as
    Markdown: tags are escaped, runs of whitespace (including the blank lines
    that mark page breaks) become one space, and `$` becomes an entity so it is
    never read as maths (which would drop it and restyle the text around it)."""
    return html.escape(re.sub(r"\s+", " ", text)).replace("$", "&#36;")


def highlight(fact: dict) -> str:
    """HTML for the evidence in context, with the quoted span marked. Every
    piece is escaped, so document text can never inject markup."""
    before = html_text(fact["ctx_before"]).lstrip()
    quote = html_text(fact["evidence_text"])
    after = html_text(fact["ctx_after"]).rstrip()
    return f"&hellip;{before}<mark>{quote}</mark>{after}&hellip;"


# One sentence pattern per relation. "{s}" is the subject, "{o}" the object(s).
# `ceo_of` is deliberately worded loosely: the extractor applies it to CFOs and
# division heads too, so claiming "chief executive" would overstate the fact.
ANSWER_TEMPLATES = {
    "acquired": "{s} acquired {o}",
    "subsidiary_of": "{s} is a subsidiary of {o}",
    "headquartered_in": "{s} is headquartered in {o}",
    "ceo_of": "{s} holds a leadership role at {o}",
    "founded_by": "{s} was founded by {o}",
    "partnered_with": "{s} partnered with {o}",
    "invested_in": "{s} invested in {o}",
    "authored_by": "{s} was written by {o}",
    "affiliated_with": "{s} is affiliated with {o}",
    "evaluated_on": "{s} was evaluated on {o}",
    "uses_method": "{s} uses {o}",
    "outperforms": "{s} outperforms {o}",
    "funded_by": "{s} was funded by {o}",
    "reports_metric": "{s} reports {o}",
}

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|>~<$&])")
MAX_PHRASE_CHARS = 140


def md_escape(text: str) -> str:
    """Escapes Markdown (and LaTeX `$`) so extracted text renders literally
    and can't turn into a link, image or formatting."""
    return _MD_SPECIAL.sub(r"\\\1", text)


def _phrase(text: str) -> str:
    text = " ".join(text.split()).strip(" ;,")
    if len(text) > MAX_PHRASE_CHARS:
        text = text[: MAX_PHRASE_CHARS - 1].rstrip() + "…"
    return text


def _join(items: list[str]) -> str:
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def generate_answer(grouped: list[dict], *, max_sentences: int = 12) -> tuple[list[str], int]:
    """Writes the answer as plain sentences, from the extracted facts only (no
    model call, so it cannot add anything that is not in the facts; it does
    inherit their mistakes). Facts about the same subject and relation are
    combined: "EGF was evaluated on QM9 and ZINC250k". Returns (sentences,
    number_of_sentences_left_out). Output is Markdown-escaped."""
    order: list[tuple[str, str]] = []
    merged: dict[tuple[str, str], dict] = {}
    for g in grouped:
        key = (norm(g["subject"]), g["relation"])
        m = merged.get(key)
        if m is None:
            m = merged[key] = {"subject": _phrase(g["subject"]), "relation": g["relation"], "objects": [], "n_docs": 1}
            order.append(key)
        obj = _phrase(g["object"])
        if obj and norm(obj) not in {norm(o) for o in m["objects"]}:
            m["objects"].append(obj)
        m["n_docs"] = max(m["n_docs"], 1 + len(g.get("also_in") or []))

    sentences = []
    for key in order[:max_sentences]:
        m = merged[key]
        template = ANSWER_TEMPLATES.get(m["relation"], "{s} (" + m["relation"] + ") {o}")
        text = template.format(s=md_escape(m["subject"]), o=_join([md_escape(o) for o in m["objects"]]))
        if m["n_docs"] > 1:
            text += f" (found in {m['n_docs']} documents)"
        sentences.append(text if text.endswith(".") else text + ".")
    return sentences, max(0, len(order) - max_sentences)
