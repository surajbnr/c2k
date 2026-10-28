"""I1: canonical_text[start:end] == evidence_text for every stored triple.
I2: no triple may be stored without a resolvable, non-empty span."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from c2k.schema import Relation, Triple
from c2k.store import Store


def _make_triple(**overrides) -> Triple:
    defaults = dict(
        triple_id="t1",
        run_id="run1",
        doc_id="doc1",
        chunk_id="doc1:0",
        subject_mention_id="t1:subj",
        object_mention_id="t1:obj",
        relation=Relation.acquired,
        evidence_start=10,
        evidence_end=24,
        evidence_text="Siemens bought",
    )
    defaults.update(overrides)
    return Triple(**defaults)


def test_i1_holds_when_span_matches_canonical(tmp_store: Store) -> None:
    text = "In 2017, Siemens bought Mentor for a lot of money."
    tmp_store.insert_document("doc1", "doc1.pdf", "sha", len(text), "train_unused")
    tmp_store.save_canonical_text("doc1", text)

    start = text.index("Siemens bought")
    triple = _make_triple(evidence_start=start, evidence_end=start + len("Siemens bought"))
    tmp_store.insert_triple(triple)

    stored = tmp_store.list_triples("run1")
    assert len(stored) == 1
    row = stored[0]
    assert text[row["evidence_start"] : row["evidence_end"]] == row["evidence_text"]


def test_i1_rejects_mismatched_span(tmp_store: Store) -> None:
    text = "In 2017, Siemens bought Mentor for a lot of money."
    tmp_store.insert_document("doc1", "doc1.pdf", "sha", len(text), "train_unused")
    tmp_store.save_canonical_text("doc1", text)

    # Correct length, wrong offset -> text at that span differs from evidence_text.
    triple = _make_triple(evidence_start=0, evidence_end=14, evidence_text="Siemens bought")
    with pytest.raises(AssertionError):
        tmp_store.insert_triple(triple)


@pytest.mark.parametrize(
    "overrides",
    [
        {"evidence_text": ""},
        {"evidence_start": 10, "evidence_end": 10},
        {"evidence_start": 20, "evidence_end": 10},
    ],
)
def test_i2_rejects_unresolvable_span_at_construction(overrides: dict) -> None:
    with pytest.raises(ValidationError):
        _make_triple(**overrides)


def test_i2_span_length_must_match_evidence_text() -> None:
    with pytest.raises(ValidationError):
        _make_triple(evidence_start=0, evidence_end=5, evidence_text="this text is way too long")
