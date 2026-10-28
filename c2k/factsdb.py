"""Turns a finished run into the flat `facts` table that the demo site
searches. Every fact is re-verified against the canonical text (I1) through
`c2k.query.search_facts`, placeholder values are dropped, and only a short
snippet of context is kept (never the whole document). Shared by the demo
export script and by the site's upload feature."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable

from c2k.query import search_facts
from c2k.store import Store

CONTEXT_CHARS = 400

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
  fact_id TEXT PRIMARY KEY, doc_id TEXT, doc_title TEXT, source_type TEXT, source_url TEXT,
  subject TEXT, subject_canonical TEXT, relation TEXT, object TEXT, object_canonical TEXT,
  evidence_text TEXT, ctx_before TEXT, ctx_after TEXT, confidence REAL);
CREATE INDEX IF NOT EXISTS facts_relation ON facts(relation);
CREATE INDEX IF NOT EXISTS facts_doc ON facts(doc_id);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

_PLACEHOLDERS = {"unknown", "n/a", "na", "none", "null", "tbd", "not specified", "not mentioned", "not applicable", "unspecified"}

# doc_id -> (title, source_type, url)
DocInfo = Callable[[str], tuple[str, str, str]]


def is_placeholder(text: str) -> bool:
    t = text.strip().lower()
    return t in _PLACEHOLDERS or (t.startswith("<") and t.endswith(">"))


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def add_run_facts(conn: sqlite3.Connection, store: Store, run_id: str, doc_info: DocInfo) -> dict[str, int]:
    """Writes the run's verified facts into `conn` (which must have the schema).
    Returns counts: stored triples, written, dropped by re-verification, and
    dropped as placeholders."""
    n_total = store._conn.execute("SELECT COUNT(*) FROM triples WHERE run_id = ?", (run_id,)).fetchone()[0]
    facts = search_facts(store, run_id)
    n_verified = len(facts)
    facts = [f for f in facts if not (is_placeholder(f.subject) or is_placeholder(f.object))]

    entity_rows = {
        r["triple_id"]: r
        for r in store._conn.execute(
            "SELECT t.triple_id, t.confidence, se.canonical_name AS sc, oe.canonical_name AS oc "
            "FROM triples t "
            "LEFT JOIN entities se ON se.entity_id = t.subject_entity_id "
            "LEFT JOIN entities oe ON oe.entity_id = t.object_entity_id "
            "WHERE t.run_id = ?",
            (run_id,),
        )
    }
    canonical_cache: dict[str, str] = {}
    for f in facts:
        text = canonical_cache.setdefault(f.doc_id, store.load_canonical_text(f.doc_id))
        title, source_type, url = doc_info(f.doc_id)
        er = entity_rows.get(f.triple_id)
        conn.execute(
            "INSERT OR REPLACE INTO facts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                f.triple_id, f.doc_id, title, source_type, url,
                f.subject, (er["sc"] if er and er["sc"] else f.subject),
                f.relation,
                f.object, (er["oc"] if er and er["oc"] else f.object),
                f.evidence_text,
                text[max(0, f.char_start - CONTEXT_CHARS) : f.char_start],
                text[f.char_end : f.char_end + CONTEXT_CHARS],
                er["confidence"] if er else None,
            ),
        )
    conn.commit()
    return {
        "stored_triples": n_total,
        "written": len(facts),
        "dropped": n_total - n_verified,
        "placeholders": n_verified - len(facts),
    }
