"""§6.3: evidence-span localization (exact / disambiguated / fuzzy / drop),
document-offset conversion, dedup, and the A0 free-form relation mapping —
all exercised with a fake LLM provider so no network or API key is needed.
"""

from __future__ import annotations

import pytest

from c2k import extract as extract_mod
from c2k import llm
from c2k.chunk import ChunkSpan
from c2k.config import (
    ChunkingConfig,
    Config,
    EvaluationConfig,
    ExtractionConfig,
    ModelConfig,
    PathsConfig,
    ResolutionConfig,
)
from c2k.extract import ExtractionResponse, ExtractionStats, FreeformExtractionResponse, locate_span
from c2k.schema import ExtractedTriple, Relation
from c2k.store import Store


def _make_config(*, constrained: bool = True) -> Config:
    return Config(
        arm="a1" if constrained else "a0",
        model=ModelConfig(provider="fake", name="fake-model", temperature=0.0),
        chunking=ChunkingConfig(),
        extraction=ExtractionConfig(constrained=constrained),
        resolution=ResolutionConfig(),
        evaluation=EvaluationConfig(),
        paths=PathsConfig(),
    )


@pytest.fixture
def fake_llm(monkeypatch):
    """`_PROVIDERS["fake"]` returns whatever the test queued via `queue.append(...)`."""
    queue: list = []

    def _provider(prompt, response_model, *, model, temperature, max_retries):
        assert queue, "test queued no fake response"
        return queue.pop(0), {"input_tokens": 1, "output_tokens": 1}

    monkeypatch.setitem(llm._PROVIDERS, "fake", _provider)
    return queue


def _seed_doc(store: Store, text: str, doc_id: str = "doc1") -> None:
    store.insert_document(doc_id, f"{doc_id}.pdf", "sha", len(text), "train_unused")
    store.save_canonical_text(doc_id, text)


def _make_chunk(doc_id: str, text: str, char_start: int = 0) -> ChunkSpan:
    return ChunkSpan(chunk_id=f"{doc_id}:0", doc_id=doc_id, idx=0, char_start=char_start, char_end=char_start + len(text), text=text)


# ---------------------------------------------------------------- locate_span


def test_locate_span_exact_single_match() -> None:
    assert locate_span("Siemens acquired Mentor Graphics.", "acquired Mentor") == (8, 23)


def test_locate_span_multiple_matches_picks_nearest_to_anchor() -> None:
    text = "X acquired Y. Elsewhere, Z acquired W."
    first = text.index("acquired")
    second = text.index("acquired", first + 1)
    assert locate_span(text, "acquired", near=second) == (second, second + len("acquired"))
    assert locate_span(text, "acquired", near=first) == (first, first + len("acquired"))


def test_locate_span_fuzzy_fallback() -> None:
    text = "Siemens acquired Mentor Graphics in 2017."
    # Model misquotes by one character; not an exact substring.
    span = locate_span(text, "Siemens aquired Mentor")
    assert span is not None
    start, end = span
    assert text[start:end].startswith("Siemens a")


def test_locate_span_gives_up_when_unrelated() -> None:
    assert locate_span("Siemens acquired Mentor Graphics.", "completely unrelated content") is None


# ---------------------------------------------------------------- dedupe


def test_dedupe_keeps_higher_confidence() -> None:
    candidates = [
        dict(subject="X", relation="acquired", object="Y", evidence_start=0, evidence_end=10, confidence=0.5),
        dict(subject="X", relation="acquired", object="Y", evidence_start=2, evidence_end=12, confidence=0.9),
    ]
    result = extract_mod._dedupe(candidates)
    assert len(result) == 1
    assert result[0]["confidence"] == 0.9


def test_dedupe_keeps_non_overlapping_as_separate() -> None:
    candidates = [
        dict(subject="X", relation="acquired", object="Y", evidence_start=0, evidence_end=10, confidence=0.5),
        dict(subject="X", relation="acquired", object="Y", evidence_start=100, evidence_end=110, confidence=0.9),
    ]
    assert len(extract_mod._dedupe(candidates)) == 2


# ---------------------------------------------------------------- extract_chunk end-to-end


def test_extract_chunk_stores_triple_with_correct_document_offsets(tmp_store: Store, fake_llm) -> None:
    text = "In 2017, Siemens acquired Mentor Graphics for billions."
    _seed_doc(tmp_store, text)
    chunk = _make_chunk("doc1", text, char_start=0)

    fake_llm.append(
        ExtractionResponse(
            triples=[
                ExtractedTriple(
                    subject="Siemens",
                    relation=Relation.acquired,
                    object="Mentor Graphics",
                    evidence_text="Siemens acquired Mentor Graphics",
                    confidence=0.95,
                )
            ]
        )
    )

    config = _make_config(constrained=True)
    llm.configure(tmp_store, config)
    tmp_store.start_run("run1", "a1", 0, "fake-model", "{}")

    stats = ExtractionStats()
    extract_mod.extract_chunk(tmp_store, config, "run1", chunk, stats)

    assert stats.n_stored == 1
    assert stats.n_span_localization_failures == 0
    triples = tmp_store.list_triples("run1")
    assert len(triples) == 1
    row = triples[0]
    assert row["relation"] == "acquired"
    expected_start = text.index("Siemens acquired Mentor Graphics")
    assert row["evidence_start"] == expected_start
    assert text[row["evidence_start"] : row["evidence_end"]] == row["evidence_text"]


def test_extract_chunk_offsets_shift_with_chunk_char_start(tmp_store: Store, fake_llm) -> None:
    prefix = "Irrelevant filler text. " * 3
    fact = "SpaceX was founded by Elon Musk."
    text = prefix + fact
    _seed_doc(tmp_store, text)
    chunk = _make_chunk("doc1", fact, char_start=len(prefix))

    fake_llm.append(
        ExtractionResponse(
            triples=[
                ExtractedTriple(
                    subject="SpaceX",
                    relation=Relation.founded_by,
                    object="Elon Musk",
                    evidence_text="SpaceX was founded by Elon Musk",
                    confidence=0.9,
                )
            ]
        )
    )
    config = _make_config(constrained=True)
    llm.configure(tmp_store, config)
    tmp_store.start_run("run1", "a1", 0, "fake-model", "{}")

    stats = ExtractionStats()
    extract_mod.extract_chunk(tmp_store, config, "run1", chunk, stats)

    row = tmp_store.list_triples("run1")[0]
    assert row["evidence_start"] == len(prefix)
    assert text[row["evidence_start"] : row["evidence_end"]] == row["evidence_text"]


def test_extract_chunk_drops_unlocatable_evidence(tmp_store: Store, fake_llm) -> None:
    text = "Siemens acquired Mentor Graphics."
    _seed_doc(tmp_store, text)
    chunk = _make_chunk("doc1", text)

    fake_llm.append(
        ExtractionResponse(
            triples=[
                ExtractedTriple(
                    subject="Nobody",
                    relation=Relation.acquired,
                    object="Nothing",
                    evidence_text="this text does not appear anywhere near here",
                    confidence=0.5,
                )
            ]
        )
    )
    config = _make_config(constrained=True)
    llm.configure(tmp_store, config)
    tmp_store.start_run("run1", "a1", 0, "fake-model", "{}")

    stats = ExtractionStats()
    extract_mod.extract_chunk(tmp_store, config, "run1", chunk, stats)

    assert stats.n_stored == 0
    assert stats.n_span_localization_failures == 1
    assert tmp_store.list_triples("run1") == []


# ---------------------------------------------------------------- A0 free-form mapping


def test_freeform_arm_maps_and_drops_relations(tmp_store: Store, fake_llm, monkeypatch) -> None:
    text = "Siemens purchased Mentor Graphics. The weather was sunny."
    _seed_doc(tmp_store, text)
    chunk = _make_chunk("doc1", text)

    fake_llm.append(
        FreeformExtractionResponse(
            triples=[
                extract_mod.FreeformExtractedTriple(
                    subject="Siemens",
                    relation="purchased",
                    object="Mentor Graphics",
                    evidence_text="Siemens purchased Mentor Graphics",
                    confidence=0.8,
                ),
                extract_mod.FreeformExtractedTriple(
                    subject="weather",
                    relation="was_sunny",
                    object="today",
                    evidence_text="The weather was sunny",
                    confidence=0.3,
                ),
            ]
        )
    )

    def fake_mapper(raw_relation, **kwargs):
        return Relation.acquired if raw_relation == "purchased" else None

    monkeypatch.setattr(extract_mod, "map_freeform_relation", fake_mapper)

    config = _make_config(constrained=False)
    llm.configure(tmp_store, config)
    tmp_store.start_run("run1", "a0", 0, "fake-model", "{}")

    stats = ExtractionStats()
    extract_mod.extract_chunk(tmp_store, config, "run1", chunk, stats)

    assert stats.n_unmapped_relations == 1
    assert stats.n_stored == 1
    row = tmp_store.list_triples("run1")[0]
    assert row["relation"] == "acquired"


# ---------------------------------------------------------------- per-chunk LLM failure resilience


def test_extract_document_survives_one_chunk_failing(tmp_store: Store, fake_llm) -> None:
    """A single chunk's LLM call exhausting retries (llm.call raises
    RuntimeError, §7) must not abort the whole document — real local models
    occasionally emit a bare text response instead of a tool call no matter
    how many retries, and a multi-hundred-chunk corpus run can't lose all
    prior progress to one such chunk. The failure is logged and counted
    (n_llm_failures), not silently dropped."""
    text_a = "Siemens acquired Mentor Graphics."
    text_b = "SpaceX was founded by Elon Musk."
    doc_id = "doc1"
    tmp_store.insert_document(doc_id, f"{doc_id}.pdf", "sha", len(text_a) + len(text_b) + 1, "train_unused")
    tmp_store.save_canonical_text(doc_id, text_a + " " + text_b)
    tmp_store.insert_chunk(f"{doc_id}:0", doc_id, 0, 0, len(text_a), text_a)
    b_start = len(text_a) + 1
    tmp_store.insert_chunk(f"{doc_id}:1", doc_id, 1, b_start, b_start + len(text_b), text_b)

    from c2k import llm as llm_mod

    fake_llm.append(
        ExtractionResponse(
            triples=[
                ExtractedTriple(
                    subject="SpaceX",
                    relation=Relation.founded_by,
                    object="Elon Musk",
                    evidence_text="SpaceX was founded by Elon Musk",
                    confidence=0.9,
                )
            ]
        )
    )
    # First call (chunk 0) must fail; second call (chunk 1) must succeed —
    # queue is empty until chunk 1, so raise until then.
    call_count = {"n": 0}

    def _provider(prompt, response_model, *, model, temperature, max_retries):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ValueError("simulated persistent tool-call failure")
        return fake_llm.pop(0), {"input_tokens": 1, "output_tokens": 1}

    llm_mod._PROVIDERS["fake"] = _provider

    config = _make_config(constrained=True)
    llm.configure(tmp_store, config)
    tmp_store.start_run("run1", "a1", 0, "fake-model", "{}")

    stats = extract_mod.extract_document(tmp_store, config, "run1", doc_id)

    assert stats.n_chunks == 2
    assert stats.n_llm_failures == 1
    assert stats.n_stored == 1
    row = tmp_store.list_triples("run1")[0]
    assert row["relation"] == "founded_by"
