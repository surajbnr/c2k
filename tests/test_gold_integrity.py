"""§15: every gold span resolves; no duplicate gold triples."""

from __future__ import annotations

from c2k.schema import GoldTriple, Relation
from c2k.store import Store


def _seed_doc(store: Store, doc_id: str = "doc1") -> str:
    text = "Siemens acquired Mentor Graphics in 2017 for a large sum."
    store.insert_document(doc_id, f"{doc_id}.pdf", "sha", len(text), "test")
    store.save_canonical_text(doc_id, text)
    return text


def test_valid_gold_passes(tmp_store: Store) -> None:
    text = _seed_doc(tmp_store)
    start = text.index("Siemens acquired Mentor Graphics")
    end = start + len("Siemens acquired Mentor Graphics")
    gold = GoldTriple(
        gold_id="g1",
        doc_id="doc1",
        subject="Siemens",
        relation=Relation.acquired,
        object="Mentor Graphics",
        evidence_start=start,
        evidence_end=end,
        annotator="sbn",
    )
    tmp_store.insert_gold_triple(gold)
    assert tmp_store.gold_integrity_report() == []


def test_out_of_bounds_span_detected(tmp_store: Store) -> None:
    text = _seed_doc(tmp_store)
    gold = GoldTriple(
        gold_id="g1",
        doc_id="doc1",
        subject="Siemens",
        relation=Relation.acquired,
        object="Mentor Graphics",
        evidence_start=0,
        evidence_end=len(text) + 50,
        annotator="sbn",
    )
    tmp_store.insert_gold_triple(gold)
    problems = tmp_store.gold_integrity_report()
    assert len(problems) == 1
    assert "out of bounds" in problems[0]


def test_duplicate_gold_detected(tmp_store: Store) -> None:
    text = _seed_doc(tmp_store)
    start = text.index("Siemens acquired Mentor Graphics")
    end = start + len("Siemens acquired Mentor Graphics")
    for gold_id in ("g1", "g2"):
        gold = GoldTriple(
            gold_id=gold_id,
            doc_id="doc1",
            subject="Siemens",
            relation=Relation.acquired,
            object="Mentor Graphics",
            evidence_start=start,
            evidence_end=end,
            annotator="sbn",
        )
        tmp_store.insert_gold_triple(gold)
    problems = tmp_store.gold_integrity_report()
    assert len(problems) == 1
    assert "duplicate" in problems[0]


def test_out_of_schema_fact_needs_no_span(tmp_store: Store) -> None:
    _seed_doc(tmp_store)
    gold = GoldTriple(
        gold_id="g1",
        doc_id="doc1",
        subject="Siemens",
        object="a strategic partner",
        out_of_schema=True,
        annotator="sbn",
    )
    tmp_store.insert_gold_triple(gold)
    assert tmp_store.gold_integrity_report() == []
