"""Private, per-visitor document upload: read a PDF, extract its facts with
Claude, and keep them in an in-memory table that only that visitor's session
can search. Nothing is written to disk beyond a temporary folder that is
deleted when the job ends, and nothing is shared with other visitors.

The extraction is the same code as for the main corpus (`c2k.extract`), so a
fact is only kept if its quoted sentence really appears in the document. One
extraction runs at a time (the LLM settings are process-wide), and every limit
below exists to keep a single upload cheap and bounded.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz

from c2k import factsdb, llm
from c2k.chunk import chunk_document, chunk_text
from c2k.config import apply_arm, load_config
from c2k.extract import extract_document
from c2k.ingest import build_canonical_text
from c2k.norm import norm
from c2k.store import Store

# Limits keep one upload cheap and bounded. Each can be raised or lowered with
# an environment variable when the site is run (a section is one piece of
# roughly 900 words that is sent to Claude; a paper averages about two per page).
MAX_BYTES = int(os.environ.get("C2K_MAX_UPLOAD_MB", 25)) * 1024 * 1024
MAX_PAGES = int(os.environ.get("C2K_MAX_PAGES", 100))
MAX_CHUNKS = int(os.environ.get("C2K_MAX_SECTIONS", 200))
MAX_DOCS_PER_SESSION = int(os.environ.get("C2K_MAX_DOCS", 3))
SECONDS_PER_CHUNK = 7  # measured: about 2 s for a short filing section, 7 s for a paper section
MIN_CHARS = 200
COST_PER_CHUNK_USD = 0.007  # measured on claude-haiku-4-5: about $0.003 per section of a short filing, $0.007 for a paper
PRICE_PER_MILLION_TOKENS = {"claude-haiku-4-5": (1.0, 5.0)}  # (input, output) USD
MERGE_SIMILARITY = 90  # names this alike (after normalising) are one entity
MAX_DISTINCT_NAMES_FOR_FUZZY_MERGE = 1500
# Trailing words that don't tell two companies apart ("NVIDIA" / "NVIDIA Corporation").
_CORPORATE_WORDS = {
    "corporation", "company", "co", "llc", "plc", "limited", "incorporated", "holdings", "group", "sa", "ag", "nv", "lp", "llp",
}

_LOCK = threading.Lock()


class UploadError(Exception):
    """A problem the visitor can act on; the message is safe to show."""


@dataclass(frozen=True)
class UploadPlan:
    name: str
    doc_id: str
    sha: str
    n_pages: int
    n_chars: int
    n_chunks: int
    est_cost_usd: float
    est_seconds: int
    text: str


@dataclass(frozen=True)
class UploadResult:
    doc_id: str
    name: str
    rows: list[tuple]
    n_chunks: int
    n_failed_chunks: int
    n_span_failures: int
    cost_usd: float | None
    seconds: float


def estimate_cost(n_chunks: int) -> float:
    return round(n_chunks * COST_PER_CHUNK_USD, 3)


def plan_upload(name: str, data: bytes) -> UploadPlan:
    """Reads the PDF locally (free) and works out what extracting it would
    involve. Raises UploadError if it is unreadable or over a limit."""
    if len(data) > MAX_BYTES:
        raise UploadError(f"That file is larger than {MAX_BYTES // (1024 * 1024)} MB.")
    safe_name = Path(name).name[:120] or "document.pdf"
    try:
        import pymupdf

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "doc.pdf"
            path.write_bytes(data)
            with pymupdf.open(path) as pdf:
                n_pages = pdf.page_count
            if n_pages > MAX_PAGES:
                raise UploadError(f"That PDF has {n_pages} pages; the limit is {MAX_PAGES}.")
            text, _spans = build_canonical_text(path)
    except UploadError:
        raise
    except Exception as exc:  # noqa: BLE001 - any parser failure is just "unreadable" to the visitor
        raise UploadError("That doesn't look like a readable PDF.") from exc
    if len(text) < MIN_CHARS:
        raise UploadError("I couldn't find readable text in that PDF (is it a scanned image?).")
    sha = hashlib.sha256(data).hexdigest()
    doc_id = f"upload_{sha[:10]}"
    n_chunks = len(chunk_text(doc_id, text))
    if n_chunks > MAX_CHUNKS:
        raise UploadError(
            f"That document is too long ({n_chunks} sections; the limit is {MAX_CHUNKS}). "
            "Try a shorter document or a part of it."
        )
    return UploadPlan(
        safe_name, doc_id, sha, n_pages, len(text), n_chunks, estimate_cost(n_chunks), n_chunks * SECONDS_PER_CHUNK, text
    )


def check_key(api_key: str) -> None:
    """Asks Anthropic for one token (costing a fraction of a cent) so a bad
    key or an empty balance is reported before any real work starts."""
    import anthropic

    model = apply_arm(load_config(), "a2").model.name
    try:
        anthropic.Anthropic(api_key=api_key).messages.create(
            model=model, max_tokens=1, messages=[{"role": "user", "content": "hi"}]
        )
    except anthropic.AuthenticationError as exc:
        raise UploadError("Anthropic rejected that key. Check that you copied the whole key.") from exc
    except anthropic.BadRequestError as exc:
        if "credit" in str(exc).lower():
            raise UploadError("That key is valid, but its account has no credit left.") from exc
        raise UploadError("Anthropic refused the test request with that key.") from exc
    except anthropic.RateLimitError as exc:
        raise UploadError("Anthropic says that key is being rate limited. Try again shortly.") from exc
    except anthropic.APIConnectionError as exc:
        raise UploadError("I couldn't reach Anthropic. Check the connection and try again.") from exc


def _merge_key(surface: str) -> str:
    """The name with case, punctuation, a leading "the" and trailing corporate
    words removed. Local to uploads; the shared `norm` is left as it is."""
    words = norm(surface).split()
    if len(words) > 1 and words[0] == "the":
        words = words[1:]
    while len(words) > 1 and words[-1] in _CORPORATE_WORDS:
        words.pop()
    return " ".join(words)


def _light_resolve(store: Store, run_id: str) -> None:
    """Merges names for the same thing within one document using string
    similarity only (no embedding model, no extra LLM calls)."""
    mentions = store.list_mentions(run_id)
    keys = sorted({_merge_key(m["surface"]) for m in mentions})
    parent = {k: k for k in keys}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    if len(keys) <= MAX_DISTINCT_NAMES_FOR_FUZZY_MERGE:
        for i, a in enumerate(keys):
            for b in keys[i + 1 :]:
                if fuzz.ratio(a, b) >= MERGE_SIMILARITY:
                    parent[find(a)] = find(b)

    clusters: dict[str, list[sqlite3.Row]] = {}
    for m in mentions:
        clusters.setdefault(find(_merge_key(m["surface"])), []).append(m)
    for root, members in clusters.items():
        counts = Counter(m["surface"] for m in members)
        canonical = max(counts, key=lambda s: (counts[s], len(s)))
        entity_id = f"{run_id}:entity:{hashlib.sha1(root.encode()).hexdigest()[:10]}"
        for m in members:
            store.set_mention_entity(m["mention_id"], entity_id)
        store.insert_entity(entity_id, run_id, canonical, len(members))
    store.sync_triple_entities_from_mentions(run_id)


def _cost_usd(store: Store, run_id: str, model: str) -> float | None:
    prices = PRICE_PER_MILLION_TOKENS.get(model)
    run = store.get_run(run_id)
    if prices is None or run is None:
        return None
    return round((run["n_input_tokens"] * prices[0] + run["n_output_tokens"] * prices[1]) / 1_000_000, 4)


def extract_upload(
    plan: UploadPlan, api_key: str | None, progress: Callable[[int, int], None] | None = None
) -> UploadResult:
    """Runs the extraction for one planned upload. `api_key` is used for this
    job only (None means the server's own ANTHROPIC_API_KEY). One job at a
    time: the LLM settings are shared by the whole process."""
    if not _LOCK.acquire(blocking=False):
        raise UploadError("Another document is being processed right now. Please try again in a minute.")
    started = time.monotonic()
    try:
        config = apply_arm(load_config(), "a2")
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "upload.sqlite", canonical_dir=Path(tmp) / "canonical")
            store.initialize()
            store.insert_document(plan.doc_id, plan.name, plan.sha, len(plan.text), "train_unused")
            store.save_canonical_text(plan.doc_id, plan.text)
            chunk_document(store, plan.doc_id)
            run_id = "upload"
            llm.configure(store, config, api_key=api_key)
            try:
                store.start_run(run_id, "a2", 0, config.model.name, config.model_dump_json())
                stats = extract_document(store, config, run_id, plan.doc_id, on_chunk=progress)
                if stats.n_chunks and stats.n_llm_failures == stats.n_chunks:
                    raise UploadError(
                        "Every request to Anthropic failed, so nothing could be extracted. "
                        "Check the key and your balance, then try again."
                    )
                _light_resolve(store, run_id)
                store.finish_run(run_id)
            finally:
                llm.reset()
            conn = new_session_db()
            factsdb.add_run_facts(conn, store, run_id, lambda doc_id: (plan.name, "upload", ""))
            rows = [tuple(r) for r in conn.execute("SELECT * FROM facts").fetchall()]
            cost = _cost_usd(store, run_id, config.model.name)
            store._conn.close()
        return UploadResult(
            plan.doc_id, plan.name, rows, stats.n_chunks, stats.n_llm_failures,
            stats.n_span_localization_failures, cost, round(time.monotonic() - started, 1),
        )
    finally:
        _LOCK.release()


# ---------------------------------------------------------------- the visitor's own table


def new_session_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    factsdb.create_schema(conn)
    return conn


def add_to_session_db(conn: sqlite3.Connection, result: UploadResult) -> None:
    conn.execute("DELETE FROM facts WHERE doc_id = ?", (result.doc_id,))
    conn.executemany("INSERT OR REPLACE INTO facts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", result.rows)
    conn.commit()


def remove_from_session_db(conn: sqlite3.Connection, doc_id: str) -> None:
    conn.execute("DELETE FROM facts WHERE doc_id = ?", (doc_id,))
    conn.commit()
