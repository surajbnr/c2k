"""Chunk -> List[Triple] (§6.3).

The model never emits character offsets — it quotes `evidence_text`
verbatim and post-processing locates it in the chunk:

  1. exact `str.find()`; if found once, done.
  2. found multiple times -> the occurrence closest to where subject and
     object both appear.
  3. not found -> `rapidfuzz.fuzz.partial_ratio_alignment`, accept if
     score >= 90, snap to the aligned window.
  4. still nothing -> drop the triple and log it (span_localization_failures).

Chunk-relative offsets become document offsets by adding `chunk.char_start`;
`store.insert_triple` asserts I1 before the row is written.

A0 (unconstrained) uses `FreeformExtractedTriple` (relation: str) and maps
the free-text predicate to the closed vocabulary post-hoc by embedding
similarity (`map_freeform_relation`) — unmapped predicates are logged and
folded into the evaluator's precision denominator as unmatchable predictions
(`RunMetrics`'s `n_unmapped_relations`), never inserted as a `Triple` (that
would violate I3: no relation string outside the closed enum).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from c2k import llm
from c2k.chunk import ChunkSpan
from c2k.config import Config
from c2k.schema import RELATION_DEFS, ExtractedTriple, Relation, Triple
from c2k.store import Store

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "extract.md"


class FreeformExtractedTriple(BaseModel):
    """A0's permissive schema (§8): same shape, `relation` unconstrained."""

    subject: str
    relation: str
    object: str
    evidence_text: str
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractionResponse(BaseModel):
    triples: list[ExtractedTriple] = Field(default_factory=list)


class FreeformExtractionResponse(BaseModel):
    triples: list[FreeformExtractedTriple] = Field(default_factory=list)


@dataclass
class ExtractionStats:
    n_chunks: int = 0
    n_extracted: int = 0
    n_stored: int = 0
    n_span_localization_failures: int = 0
    n_deduplicated: int = 0
    n_unmapped_relations: int = 0
    n_llm_failures: int = 0


def _relations_block() -> str:
    lines = []
    for name, entry in RELATION_DEFS.items():
        exemplars = "; ".join(entry["exemplars"])
        lines.append(f"- {name}: {entry['definition']} (e.g. {exemplars})")
    return "\n".join(lines)


def build_extract_prompt(chunk_text: str) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.format(relations_block=_relations_block(), chunk_text=chunk_text)


def locate_span(
    haystack: str, needle: str, *, near: int | None = None, min_fuzzy_score: int = 90
) -> tuple[int, int] | None:
    """§6.3 steps 1-4, generalised for locating any substring (evidence text
    or a subject/object surface) inside `haystack`. Returns a (start, end)
    span relative to `haystack`, or None if it can't be resolved."""
    if not needle:
        return None

    occurrences = []
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            break
        occurrences.append(idx)
        start = idx + 1

    if len(occurrences) == 1:
        return occurrences[0], occurrences[0] + len(needle)
    if len(occurrences) > 1:
        if near is None:
            return occurrences[0], occurrences[0] + len(needle)
        best = min(occurrences, key=lambda i: abs(i - near))
        return best, best + len(needle)

    alignment = fuzz.partial_ratio_alignment(needle, haystack)
    if alignment is not None and alignment.score >= min_fuzzy_score:
        return alignment.dest_start, alignment.dest_end
    return None


def _midpoint(span: tuple[int, int] | None) -> int | None:
    return None if span is None else (span[0] + span[1]) // 2


def _log_span_failure(run_id: str, chunk: ChunkSpan, raw_subject: str, raw_relation: str, raw_object: str, raw_evidence: str) -> None:
    print(
        f"[span_localization_failure] run={run_id} chunk={chunk.chunk_id} "
        f"relation={raw_relation!r} evidence_text={raw_evidence!r} chunk_text={chunk.text!r}"
    )


def _dedupe(candidates: list[dict]) -> list[dict]:
    """§6.3: identical (subject, relation, object) with overlapping spans
    collapse to one, keeping the higher confidence."""
    kept: list[dict] = []
    for cand in candidates:
        merged = False
        for existing in kept:
            if (
                existing["subject"] == cand["subject"]
                and existing["relation"] == cand["relation"]
                and existing["object"] == cand["object"]
                and existing["evidence_start"] < cand["evidence_end"]
                and cand["evidence_start"] < existing["evidence_end"]
            ):
                if cand["confidence"] > existing["confidence"]:
                    existing.update(cand)
                merged = True
                break
        if not merged:
            kept.append(cand)
    return kept


_EMBEDDER = None
_RELATION_EMBEDDINGS = None


def _get_embedder(model_name: str):
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER = SentenceTransformer(model_name)
    return _EMBEDDER


def map_freeform_relation(
    raw_relation: str, *, embedding_model: str = "BAAI/bge-base-en-v1.5", threshold: float = 0.55
) -> Relation | None:
    """A0 post-hoc mapping: embed the free-text predicate and the definition
    + exemplars of every closed relation, take the closest by cosine
    similarity, and reject the mapping below `threshold` (an unmapped
    predicate — §8's 'unmapped predicates counting as misses')."""
    import numpy as np

    global _RELATION_EMBEDDINGS
    embedder = _get_embedder(embedding_model)
    if _RELATION_EMBEDDINGS is None:
        texts = [
            f"{name}: {entry['definition']} " + " ".join(entry["exemplars"])
            for name, entry in RELATION_DEFS.items()
        ]
        _RELATION_EMBEDDINGS = (
            list(RELATION_DEFS.keys()),
            np.asarray(embedder.encode(texts, normalize_embeddings=True)),
        )
    names, matrix = _RELATION_EMBEDDINGS
    query = np.asarray(embedder.encode([raw_relation], normalize_embeddings=True))[0]
    sims = matrix @ query
    best_idx = int(sims.argmax())
    if sims[best_idx] < threshold:
        return None
    return Relation(names[best_idx])


def extract_chunk(
    store: Store,
    config: Config,
    run_id: str,
    chunk: ChunkSpan,
    stats: ExtractionStats,
) -> None:
    prompt = build_extract_prompt(chunk.text)
    stats.n_chunks += 1

    candidates: list[dict] = []
    if config.extraction.constrained:
        response = llm.call(prompt, ExtractionResponse, run_id=run_id, purpose="extract")
        for t in response.triples:
            candidates.append(
                dict(
                    subject=t.subject,
                    relation=t.relation.value,
                    object=t.object,
                    evidence_text=t.evidence_text,
                    confidence=t.confidence,
                )
            )
    else:
        response = llm.call(prompt, FreeformExtractionResponse, run_id=run_id, purpose="extract")
        for t in response.triples:
            mapped = map_freeform_relation(t.relation, embedding_model=config.resolution.embedding_model)
            if mapped is None:
                stats.n_unmapped_relations += 1
                _log_span_failure(run_id, chunk, t.subject, t.relation, t.object, t.evidence_text)
                continue
            candidates.append(
                dict(
                    subject=t.subject,
                    relation=mapped.value,
                    object=t.object,
                    evidence_text=t.evidence_text,
                    confidence=t.confidence,
                )
            )

    stats.n_extracted += len(candidates)

    located: list[dict] = []
    for cand in candidates:
        span = locate_span(chunk.text, cand["evidence_text"])
        if span is None:
            stats.n_span_localization_failures += 1
            _log_span_failure(run_id, chunk, cand["subject"], cand["relation"], cand["object"], cand["evidence_text"])
            continue
        ev_start, ev_end = span
        subj_near = _midpoint(locate_span(chunk.text, cand["subject"], near=ev_start))
        obj_near = _midpoint(locate_span(chunk.text, cand["object"], near=ev_start))
        near = ev_start
        if subj_near is not None and obj_near is not None:
            near = (subj_near + obj_near) // 2
        # Re-resolve evidence against the subject/object-informed anchor when
        # the exact text occurred more than once in the chunk (§6.3 step 2).
        span = locate_span(chunk.text, cand["evidence_text"], near=near) or span
        ev_start, ev_end = span

        subject_span = locate_span(chunk.text, cand["subject"], near=ev_start) or (ev_start, ev_start)
        object_span = locate_span(chunk.text, cand["object"], near=ev_start) or (ev_start, ev_start)

        located.append(
            {
                **cand,
                "evidence_start": chunk.char_start + ev_start,
                "evidence_end": chunk.char_start + ev_end,
                "evidence_text": chunk.text[ev_start:ev_end],
                "subject_start": chunk.char_start + subject_span[0],
                "subject_end": chunk.char_start + subject_span[1],
                "object_start": chunk.char_start + object_span[0],
                "object_end": chunk.char_start + object_span[1],
            }
        )

    deduped = _dedupe(located)
    stats.n_deduplicated += len(located) - len(deduped)

    for cand in deduped:
        triple_id = f"{run_id}:{uuid.uuid4().hex[:12]}"
        subj_mention_id = f"{triple_id}:subj"
        obj_mention_id = f"{triple_id}:obj"
        store.insert_mention(
            subj_mention_id, run_id, chunk.doc_id, cand["subject"], cand["subject_start"], cand["subject_end"]
        )
        store.insert_mention(
            obj_mention_id, run_id, chunk.doc_id, cand["object"], cand["object_start"], cand["object_end"]
        )
        triple = Triple(
            triple_id=triple_id,
            run_id=run_id,
            doc_id=chunk.doc_id,
            chunk_id=chunk.chunk_id,
            subject_mention_id=subj_mention_id,
            object_mention_id=obj_mention_id,
            relation=Relation(cand["relation"]),
            evidence_start=cand["evidence_start"],
            evidence_end=cand["evidence_end"],
            evidence_text=cand["evidence_text"],
            confidence=cand["confidence"],
        )
        store.insert_triple(triple)
        stats.n_stored += 1


def extract_document(
    store: Store, config: Config, run_id: str, doc_id: str, on_chunk: Callable[[int, int], None] | None = None
) -> ExtractionStats:
    """`on_chunk(done, total)`, if given, is called after each chunk (used to
    show progress); it does not affect the result."""
    stats = ExtractionStats()
    rows = store.list_chunks(doc_id)
    for done, row in enumerate(rows, start=1):
        chunk = ChunkSpan(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            idx=row["idx"],
            char_start=row["char_start"],
            char_end=row["char_end"],
            text=row["text"],
        )
        try:
            extract_chunk(store, config, run_id, chunk, stats)
        except RuntimeError as exc:
            # llm.call() raises on persistent schema failure by design (§7) —
            # that's correct at the single-call level, but a multi-hundred-
            # chunk corpus run must not lose all prior progress because one
            # chunk's call never resolved (a real failure mode with smaller
            # local models, which occasionally emit a bare text response
            # instead of a tool call no matter how many retries). Logged and
            # counted here, same as any other per-chunk extraction failure,
            # rather than silently ignored.
            stats.n_llm_failures += 1
            print(f"[llm_failure] run={run_id} doc={doc_id} chunk={chunk.chunk_id} error={exc}")
        if on_chunk is not None:
            on_chunk(done, len(rows))
    return stats


def extract_corpus(store: Store, config: Config, run_id: str, doc_ids: list[str] | None = None) -> ExtractionStats:
    total = ExtractionStats()
    ids = doc_ids if doc_ids is not None else [r["doc_id"] for r in store.list_documents()]
    for doc_id in ids:
        stats = extract_document(store, config, run_id, doc_id)
        total.n_chunks += stats.n_chunks
        total.n_extracted += stats.n_extracted
        total.n_stored += stats.n_stored
        total.n_span_localization_failures += stats.n_span_localization_failures
        total.n_deduplicated += stats.n_deduplicated
        total.n_unmapped_relations += stats.n_unmapped_relations
        total.n_llm_failures += stats.n_llm_failures
    return total
