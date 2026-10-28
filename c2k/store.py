"""SQLite repository. Every module that touches the database goes through
`Store` — no other module imports `sqlite3` (mirrors I4's discipline for the
LLM layer, checked by convention + `tests/test_no_direct_sdk.py`'s sibling
scan is not required by the spec, but the pattern is the same).

Invariant I1 (`canonical_text[start:end] == evidence_text`) is asserted in
`insert_triple` before the row is written.

Invariant I5 (test split is read-only outside `c2k eval --split test`) is
enforced structurally here: `load_gold(Split.TEST, ...)` raises
`TestSplitLocked` unless called with a `TestUnlock` instance. `TestUnlock` is
constructed only inside `cli.py`'s `eval` command when the user literally
typed `--split test`; no tuning or sweep code path imports it.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterator

from c2k.schema import GoldTriple, Triple

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  doc_id TEXT PRIMARY KEY, path TEXT, sha256 TEXT,
  n_chars INTEGER, split TEXT, ingested_at TEXT);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id TEXT PRIMARY KEY, doc_id TEXT, idx INTEGER,
  char_start INTEGER, char_end INTEGER, text TEXT);

CREATE TABLE IF NOT EXISTS mentions (
  mention_id TEXT PRIMARY KEY, run_id TEXT, doc_id TEXT, surface TEXT,
  char_start INTEGER, char_end INTEGER, entity_id TEXT);

CREATE TABLE IF NOT EXISTS entities (
  entity_id TEXT PRIMARY KEY, run_id TEXT, canonical_name TEXT,
  n_mentions INTEGER);

CREATE TABLE IF NOT EXISTS triples (
  triple_id TEXT PRIMARY KEY, run_id TEXT, doc_id TEXT, chunk_id TEXT,
  subject_mention_id TEXT, subject_entity_id TEXT,
  relation TEXT NOT NULL,
  object_mention_id TEXT, object_entity_id TEXT,
  evidence_start INTEGER NOT NULL, evidence_end INTEGER NOT NULL,
  evidence_text TEXT NOT NULL, confidence REAL);

CREATE TABLE IF NOT EXISTS gold_triples (
  gold_id TEXT PRIMARY KEY, doc_id TEXT, subject TEXT, relation TEXT,
  object TEXT, evidence_start INTEGER, evidence_end INTEGER,
  out_of_schema INTEGER DEFAULT 0, annotator TEXT);

CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, arm TEXT, repeat_idx INTEGER, model TEXT,
  config_json TEXT, started_at TEXT, finished_at TEXT,
  n_calls INTEGER, n_input_tokens INTEGER, n_output_tokens INTEGER,
  wall_seconds REAL);

CREATE TABLE IF NOT EXISTS llm_cache (
  key TEXT PRIMARY KEY, model TEXT, response_json TEXT, created_at TEXT);

CREATE TABLE IF NOT EXISTS entailment_judgements (
  triple_id TEXT, run_id TEXT, judge TEXT, verdict INTEGER,
  PRIMARY KEY (triple_id, judge));

CREATE TABLE IF NOT EXISTS run_stats (
  run_id TEXT PRIMARY KEY,
  n_chunks INTEGER, n_extracted INTEGER, n_stored INTEGER,
  n_span_localization_failures INTEGER, n_deduplicated INTEGER,
  n_unmapped_relations INTEGER, n_llm_failures INTEGER,
  n_mentions INTEGER, n_candidate_pairs INTEGER, n_accepted_direct INTEGER,
  n_rejected INTEGER, n_deferred INTEGER, n_adjudicated_same INTEGER,
  n_entities INTEGER);
"""

_RUN_STATS_COLUMNS = (
    "n_chunks", "n_extracted", "n_stored", "n_span_localization_failures",
    "n_deduplicated", "n_unmapped_relations", "n_llm_failures", "n_mentions",
    "n_candidate_pairs", "n_accepted_direct", "n_rejected", "n_deferred",
    "n_adjudicated_same", "n_entities",
)


class Split(str, Enum):
    TRAIN_UNUSED = "train_unused"
    TUNE = "tune"
    TEST = "test"


class TestSplitLocked(RuntimeError):
    """Raised when `load_gold(Split.TEST)` is called without an explicit
    TestUnlock. See I5: the test split is read-only outside `c2k eval
    --split test`."""


@dataclass(frozen=True)
class TestUnlock:
    """Proof that a human explicitly requested `--split test`. Construct this
    only where that CLI flag is parsed (`cli.py`'s `eval` command) — never in
    `tune.py`, threshold-sweep code, or any code path that also touches the
    tune split in the same run."""

    reason: str = "cli --split test"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, db_path: Path | str, canonical_dir: Path | str = "data/canonical"):
        self.db_path = Path(db_path)
        self.canonical_dir = Path(canonical_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row

    def initialize(self) -> None:
        self._conn.executescript(_SCHEMA)
        # Lightweight migration for databases created before a column was
        # added to run_stats — CREATE TABLE IF NOT EXISTS above is a no-op
        # against an already-existing table, so new columns need an explicit
        # ALTER TABLE. Idempotent: skipped once the column is present.
        existing = {row["name"] for row in self._conn.execute("PRAGMA table_info(run_stats)")}
        for col in _RUN_STATS_COLUMNS:
            if col not in existing:
                self._conn.execute(f"ALTER TABLE run_stats ADD COLUMN {col} INTEGER")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ---------------------------------------------------------------- canonical text

    def canonical_path(self, doc_id: str) -> Path:
        return self.canonical_dir / f"{doc_id}.txt"

    def load_canonical_text(self, doc_id: str) -> str:
        return self.canonical_path(doc_id).read_text(encoding="utf-8")

    def save_canonical_text(self, doc_id: str, text: str) -> None:
        self.canonical_dir.mkdir(parents=True, exist_ok=True)
        self.canonical_path(doc_id).write_text(text, encoding="utf-8")

    # ---------------------------------------------------------------- documents / chunks

    def insert_document(
        self,
        doc_id: str,
        path: str,
        sha256: str,
        n_chars: int,
        split: Split | str,
    ) -> None:
        split_value = split.value if isinstance(split, Split) else split
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO documents "
                "(doc_id, path, sha256, n_chars, split, ingested_at) VALUES (?,?,?,?,?,?)",
                (doc_id, path, sha256, n_chars, split_value, _now()),
            )

    def get_document(self, doc_id: str) -> sqlite3.Row | None:
        cur = self._conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
        return cur.fetchone()

    def list_documents(self) -> list[sqlite3.Row]:
        return self._conn.execute("SELECT * FROM documents").fetchall()

    def insert_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        idx: int,
        char_start: int,
        char_end: int,
        text: str,
    ) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO chunks "
                "(chunk_id, doc_id, idx, char_start, char_end, text) VALUES (?,?,?,?,?,?)",
                (chunk_id, doc_id, idx, char_start, char_end, text),
            )

    def list_chunks(self, doc_id: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM chunks WHERE doc_id = ? ORDER BY idx", (doc_id,)
        ).fetchall()

    # ---------------------------------------------------------------- mentions / entities

    def insert_mention(
        self,
        mention_id: str,
        run_id: str,
        doc_id: str,
        surface: str,
        char_start: int,
        char_end: int,
        entity_id: str | None = None,
    ) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO mentions "
                "(mention_id, run_id, doc_id, surface, char_start, char_end, entity_id) "
                "VALUES (?,?,?,?,?,?,?)",
                (mention_id, run_id, doc_id, surface, char_start, char_end, entity_id),
            )

    def insert_entity(
        self, entity_id: str, run_id: str, canonical_name: str, n_mentions: int
    ) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO entities "
                "(entity_id, run_id, canonical_name, n_mentions) VALUES (?,?,?,?)",
                (entity_id, run_id, canonical_name, n_mentions),
            )

    def list_mentions(self, run_id: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM mentions WHERE run_id = ? ORDER BY doc_id, char_start", (run_id,)
        ).fetchall()

    def set_mention_entity(self, mention_id: str, entity_id: str) -> None:
        with self._cursor() as cur:
            cur.execute("UPDATE mentions SET entity_id = ? WHERE mention_id = ?", (entity_id, mention_id))

    def set_triple_entities(
        self, triple_id: str, subject_entity_id: str | None, object_entity_id: str | None
    ) -> None:
        with self._cursor() as cur:
            cur.execute(
                "UPDATE triples SET subject_entity_id = ?, object_entity_id = ? WHERE triple_id = ?",
                (subject_entity_id, object_entity_id, triple_id),
            )

    def sync_triple_entities_from_mentions(self, run_id: str) -> None:
        """After clustering sets `mentions.entity_id`, propagate those ids
        onto the triples that reference each mention as subject/object."""
        with self._cursor() as cur:
            cur.execute(
                "UPDATE triples SET subject_entity_id = ("
                "  SELECT entity_id FROM mentions WHERE mention_id = triples.subject_mention_id"
                ") WHERE run_id = ?",
                (run_id,),
            )
            cur.execute(
                "UPDATE triples SET object_entity_id = ("
                "  SELECT entity_id FROM mentions WHERE mention_id = triples.object_mention_id"
                ") WHERE run_id = ?",
                (run_id,),
            )

    # ---------------------------------------------------------------- triples

    def insert_triple(self, triple: Triple) -> None:
        """Asserts I1 before writing: the stored evidence span must reproduce
        the stored evidence text exactly against the canonical stream."""
        canonical = self.load_canonical_text(triple.doc_id)
        actual = canonical[triple.evidence_start : triple.evidence_end]
        if actual != triple.evidence_text:
            raise AssertionError(
                f"I1 violated for {triple.triple_id}: canonical[{triple.evidence_start}:"
                f"{triple.evidence_end}] = {actual!r} != evidence_text = {triple.evidence_text!r}"
            )
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO triples "
                "(triple_id, run_id, doc_id, chunk_id, subject_mention_id, subject_entity_id, "
                " relation, object_mention_id, object_entity_id, evidence_start, evidence_end, "
                " evidence_text, confidence) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    triple.triple_id,
                    triple.run_id,
                    triple.doc_id,
                    triple.chunk_id,
                    triple.subject_mention_id,
                    triple.subject_entity_id,
                    triple.relation.value,
                    triple.object_mention_id,
                    triple.object_entity_id,
                    triple.evidence_start,
                    triple.evidence_end,
                    triple.evidence_text,
                    triple.confidence,
                ),
            )

    def list_triples(self, run_id: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM triples WHERE run_id = ?", (run_id,)
        ).fetchall()

    def list_triples_with_surface(self, run_id: str) -> list[sqlite3.Row]:
        """Triples joined to their mentions' surface text — what the
        evaluator needs to compare against gold (subject/object as strings)."""
        return self._conn.execute(
            "SELECT t.*, sm.surface AS subject_surface, om.surface AS object_surface "
            "FROM triples t "
            "JOIN mentions sm ON sm.mention_id = t.subject_mention_id "
            "JOIN mentions om ON om.mention_id = t.object_mention_id "
            "WHERE t.run_id = ?",
            (run_id,),
        ).fetchall()

    # ---------------------------------------------------------------- gold

    def insert_gold_triple(self, gold: GoldTriple) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO gold_triples "
                "(gold_id, doc_id, subject, relation, object, evidence_start, evidence_end, "
                " out_of_schema, annotator) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    gold.gold_id,
                    gold.doc_id,
                    gold.subject,
                    gold.relation.value if gold.relation is not None else None,
                    gold.object,
                    gold.evidence_start,
                    gold.evidence_end,
                    int(gold.out_of_schema),
                    gold.annotator,
                ),
            )

    def load_gold(
        self, split: Split, *, unlock: TestUnlock | None = None
    ) -> list[sqlite3.Row]:
        """I5: reading the test split requires an explicit TestUnlock. Every
        tuning / threshold-sweep path calls this with Split.TUNE and never
        constructs a TestUnlock, so it structurally cannot reach the test
        rows."""
        if split is Split.TEST and unlock is None:
            raise TestSplitLocked(
                "Split.TEST requires an explicit TestUnlock — only `c2k eval "
                "--split test` may construct one (I5)."
            )
        return self._conn.execute(
            "SELECT g.* FROM gold_triples g "
            "JOIN documents d ON d.doc_id = g.doc_id "
            "WHERE d.split = ?",
            (split.value,),
        ).fetchall()

    def sync_gold_from_jsonl(self, gold_dir: Path | str) -> int:
        """Loads every `data/gold/{doc_id}.jsonl` file into `gold_triples`.
        The JSONL is the durable source of truth (§10); this keeps the
        queryable table `load_gold` reads from in sync with it. Returns the
        number of rows synced."""
        n = 0
        for path in sorted(Path(gold_dir).glob("*.jsonl")):
            doc_id = path.stem
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                raw.setdefault("gold_id", f"{doc_id}:{i}")
                raw.setdefault("doc_id", doc_id)
                self.insert_gold_triple(GoldTriple.model_validate(raw))
                n += 1
        return n

    def gold_integrity_report(self) -> list[str]:
        """§15 `test_gold_integrity.py`: every gold span resolves against its
        document's canonical text, and no two gold rows are duplicates.
        Returns a list of problem descriptions; empty means clean."""
        problems: list[str] = []
        seen: set[tuple] = set()
        for r in self._conn.execute("SELECT * FROM gold_triples").fetchall():
            key = (r["doc_id"], r["subject"], r["relation"], r["object"], r["evidence_start"], r["evidence_end"])
            if key in seen:
                problems.append(f"duplicate gold triple: {r['gold_id']}")
            seen.add(key)
            if r["out_of_schema"]:
                continue
            if r["evidence_start"] is None or r["evidence_end"] is None:
                problems.append(f"{r['gold_id']}: in-schema fact missing a span")
                continue
            if not self.canonical_path(r["doc_id"]).exists():
                problems.append(f"{r['gold_id']}: no canonical text for doc {r['doc_id']}")
                continue
            text = self.load_canonical_text(r["doc_id"])
            if not (0 <= r["evidence_start"] < r["evidence_end"] <= len(text)):
                problems.append(f"{r['gold_id']}: span [{r['evidence_start']}:{r['evidence_end']}] out of bounds")
        return problems

    # ---------------------------------------------------------------- runs

    def start_run(self, run_id: str, arm: str, repeat_idx: int, model: str, config_json: str) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO runs "
                "(run_id, arm, repeat_idx, model, config_json, started_at, finished_at, "
                " n_calls, n_input_tokens, n_output_tokens, wall_seconds) "
                "VALUES (?,?,?,?,?,?,NULL,0,0,0,0.0)",
                (run_id, arm, repeat_idx, model, config_json, _now()),
            )

    def record_call(self, run_id: str, n_input_tokens: int, n_output_tokens: int, wall_seconds: float) -> None:
        with self._cursor() as cur:
            cur.execute(
                "UPDATE runs SET n_calls = n_calls + 1, "
                "n_input_tokens = n_input_tokens + ?, "
                "n_output_tokens = n_output_tokens + ?, "
                "wall_seconds = wall_seconds + ? WHERE run_id = ?",
                (n_input_tokens, n_output_tokens, wall_seconds, run_id),
            )

    def finish_run(self, run_id: str) -> None:
        with self._cursor() as cur:
            cur.execute("UPDATE runs SET finished_at = ? WHERE run_id = ?", (_now(), run_id))

    def get_run(self, run_id: str) -> sqlite3.Row | None:
        return self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()

    def get_run_repeat_idx(self, run_id: str) -> int:
        row = self.get_run(run_id)
        if row is None:
            raise KeyError(f"no such run: {run_id}")
        return row["repeat_idx"]

    # ---------------------------------------------------------------- run stats
    #
    # In-process stats (`ExtractionStats`, `ResolveStats`) don't survive past
    # the CLI process that computed them, so `c2k eval` run later — possibly
    # in a fresh process — has nothing to report span-localization-failure
    # rate, adjudication rate, etc. against. `run_stats` persists them,
    # keyed by `run_id`. Extraction and resolution each write only their own
    # columns (via `ON CONFLICT ... DO UPDATE SET <their columns>`), so
    # running extract then resolve against the same run_id merges into one
    # row instead of one stage clobbering the other's columns.

    def record_extraction_stats(
        self,
        run_id: str,
        *,
        n_chunks: int,
        n_extracted: int,
        n_stored: int,
        n_span_localization_failures: int,
        n_deduplicated: int,
        n_unmapped_relations: int,
        n_llm_failures: int = 0,
    ) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO run_stats "
                "(run_id, n_chunks, n_extracted, n_stored, "
                " n_span_localization_failures, n_deduplicated, n_unmapped_relations, n_llm_failures) "
                "VALUES (?,?,?,?,?,?,?,?) "
                "ON CONFLICT(run_id) DO UPDATE SET "
                " n_chunks=excluded.n_chunks, n_extracted=excluded.n_extracted, "
                " n_stored=excluded.n_stored, "
                " n_span_localization_failures=excluded.n_span_localization_failures, "
                " n_deduplicated=excluded.n_deduplicated, "
                " n_unmapped_relations=excluded.n_unmapped_relations, "
                " n_llm_failures=excluded.n_llm_failures",
                (
                    run_id, n_chunks, n_extracted, n_stored,
                    n_span_localization_failures, n_deduplicated, n_unmapped_relations, n_llm_failures,
                ),
            )

    def record_resolution_stats(
        self,
        run_id: str,
        *,
        n_mentions: int,
        n_candidate_pairs: int,
        n_accepted_direct: int,
        n_rejected: int,
        n_deferred: int,
        n_adjudicated_same: int,
        n_entities: int,
    ) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO run_stats "
                "(run_id, n_mentions, n_candidate_pairs, n_accepted_direct, "
                " n_rejected, n_deferred, n_adjudicated_same, n_entities) "
                "VALUES (?,?,?,?,?,?,?,?) "
                "ON CONFLICT(run_id) DO UPDATE SET "
                " n_mentions=excluded.n_mentions, n_candidate_pairs=excluded.n_candidate_pairs, "
                " n_accepted_direct=excluded.n_accepted_direct, n_rejected=excluded.n_rejected, "
                " n_deferred=excluded.n_deferred, n_adjudicated_same=excluded.n_adjudicated_same, "
                " n_entities=excluded.n_entities",
                (
                    run_id, n_mentions, n_candidate_pairs, n_accepted_direct,
                    n_rejected, n_deferred, n_adjudicated_same, n_entities,
                ),
            )

    def get_run_stats(self, run_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM run_stats WHERE run_id = ?", (run_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    # ---------------------------------------------------------------- llm cache

    def cache_get(self, key: str) -> dict | None:
        row = self._conn.execute(
            "SELECT response_json FROM llm_cache WHERE key = ?", (key,)
        ).fetchone()
        return json.loads(row["response_json"]) if row is not None else None

    def cache_put(self, key: str, model: str, response_json: dict) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO llm_cache (key, model, response_json, created_at) "
                "VALUES (?,?,?,?)",
                (key, model, json.dumps(response_json), _now()),
            )

    # ---------------------------------------------------------------- entailment

    def insert_entailment_judgement(
        self, triple_id: str, run_id: str, judge: str, verdict: bool
    ) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO entailment_judgements "
                "(triple_id, run_id, judge, verdict) VALUES (?,?,?,?)",
                (triple_id, run_id, judge, int(verdict)),
            )

    def get_entailment_judgements(self, run_id: str, judge: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM entailment_judgements WHERE run_id = ? AND judge = ?",
            (run_id, judge),
        ).fetchall()
