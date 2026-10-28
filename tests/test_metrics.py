"""§15: hand-computed P/R/F1 on a small fixture — "worth more than any other
test here" per SPEC.md, since a wrong evaluator makes every downstream number
wrong and nothing else would catch it.

Fixture (6 predicted, 5 gold: 4 in-schema + 1 out-of-schema):

  p0 (Siemens Corp, acquired, Mentor Graphics Inc)   -> matches g0 (suffix-stripped exact)
  p1 (Acme Corp, headquartered_in, Springfield)      -> matches g1 (suffix-stripped exact)
  p2 (SpaceX, founded_by, Elon Musk)                 -> matches g2 (exact)
  p3 (Random Co, invested_in, Random Fund)           -> no gold has this relation -> FP
  p4 (Alice, ceo_of, Zebra Holdings)                 -> object far from g3's "Widget Co" -> FP
  p5 (SpaceX, founded_by, Elon Musk)                 -> duplicate of p2, competes for g2 -> FP
                                                         (one-to-one: only one of p2/p5 matches)
  g4 is out_of_schema (no relation), counted only in schema recall ceiling.

matched = 3 => precision = 3/6 = 0.5, recall = 3/4 = 0.75, f1 = 0.6
"""

from __future__ import annotations

from c2k.eval.match import EvalTriple
from c2k.eval.metrics import (
    load_predicted,
    PRF1,
    adjudication_rate,
    er_pairwise_f1,
    evaluate_run,
    gold_clusters_for_mentions,
    hallucination_rate,
    precision_recall_f1,
    predicted_clusters_from_store,
    schema_recall_ceiling,
    span_localization_failure_rate,
    triple_prf1,
)
from c2k.schema import GoldTriple, Relation, Triple
from c2k.store import Split

PREDICTED = [
    EvalTriple("p0", "Siemens Corp", Relation.acquired, "Mentor Graphics Inc"),
    EvalTriple("p1", "Acme Corp", Relation.headquartered_in, "Springfield"),
    EvalTriple("p2", "SpaceX", Relation.founded_by, "Elon Musk"),
    EvalTriple("p3", "Random Co", Relation.invested_in, "Random Fund"),
    EvalTriple("p4", "Alice", Relation.ceo_of, "Zebra Holdings"),
    EvalTriple("p5", "SpaceX", Relation.founded_by, "Elon Musk"),
]

GOLD_IN_SCHEMA = [
    EvalTriple("g0", "Siemens", Relation.acquired, "Mentor Graphics"),
    EvalTriple("g1", "Acme", Relation.headquartered_in, "Springfield"),
    EvalTriple("g2", "SpaceX", Relation.founded_by, "Elon Musk"),
    EvalTriple("g3", "Alice", Relation.ceo_of, "Widget Co"),
]
N_GOLD_TOTAL = 5  # 4 in-schema above + 1 out-of-schema


def test_precision_recall_f1_hand_computed() -> None:
    assert precision_recall_f1(3, 6, 4) == PRF1(0.5, 0.75, 0.6)


def test_precision_recall_f1_edge_cases() -> None:
    assert precision_recall_f1(0, 0, 5) == PRF1(0.0, 0.0, 0.0)
    assert precision_recall_f1(0, 5, 0) == PRF1(0.0, 0.0, 0.0)


def test_triple_prf1_on_fixture() -> None:
    result = triple_prf1(PREDICTED, GOLD_IN_SCHEMA, entity_threshold=0.85)
    assert result == PRF1(0.5, 0.75, 0.6)


def test_schema_recall_ceiling() -> None:
    n_out_of_schema = N_GOLD_TOTAL - len(GOLD_IN_SCHEMA)
    assert n_out_of_schema == 1
    assert schema_recall_ceiling(n_out_of_schema, N_GOLD_TOTAL) == 0.8


def test_hallucination_rate() -> None:
    assert hallucination_rate([True, True, False, True]) == 0.25
    assert hallucination_rate([]) is None


def test_span_localization_failure_rate() -> None:
    assert span_localization_failure_rate(2, 8) == 0.2
    assert span_localization_failure_rate(0, 0) is None


def test_adjudication_rate() -> None:
    assert adjudication_rate(3, 50) == 0.06
    assert adjudication_rate(0, 0) is None


def test_er_pairwise_f1_hand_computed() -> None:
    # 4 mentions: m1,m2 truly co-referent; m3,m4 truly distinct.
    # predicted clusters everything into one blob -> over-merges m3/m4 with the rest.
    predicted = {"m1": "e", "m2": "e", "m3": "e", "m4": "e"}
    gold = {"m1": "gA", "m2": "gA", "m3": "gB", "m4": "gC"}
    # pairs: (m1,m2) pred=same gold=same -> TP
    #        (m1,m3) pred=same gold=diff -> FP
    #        (m1,m4) pred=same gold=diff -> FP
    #        (m2,m3) pred=same gold=diff -> FP
    #        (m2,m4) pred=same gold=diff -> FP
    #        (m3,m4) pred=same gold=diff -> FP
    # n_pred_pos=6, n_gold_pos=1, n_matched=1
    result = er_pairwise_f1(predicted, gold)
    expected = precision_recall_f1(1, 6, 1)
    assert result == expected
    assert round(result.precision, 4) == round(1 / 6, 4)
    assert result.recall == 1.0


def test_er_pairwise_f1_none_when_no_clustering() -> None:
    assert er_pairwise_f1({}, {"m1": "g1"}) is None
    assert er_pairwise_f1({"m1": "e1"}, {}) is None


def test_triple_prf1_unmatchable_predicted_counts_as_miss() -> None:
    """§8: A0's free-form predicates that failed post-hoc relation mapping
    never became a `Triple` (I3), so they can't appear in `predicted` — but
    they still count against precision. Same fixture as above, plus 2
    unmatchable predictions: matched stays 3, but the denominator grows from
    6 to 8 -> precision 3/8=0.375, recall untouched at 0.75."""
    result = triple_prf1(PREDICTED, GOLD_IN_SCHEMA, entity_threshold=0.85, n_unmatchable_predicted=2)
    assert result.precision == 0.375
    assert result.recall == 0.75
    assert round(result.f1, 4) == 0.5


def test_gold_clusters_for_mentions_hand_computed() -> None:
    """§9.2 ER pairwise F1: the gold JSONL format carries no coreference
    labels, so "gold-same" is derived by nearest-match to a gold subject/
    object string within the same document. m3 matches nothing above
    threshold and is dropped, not force-clustered."""
    mention_rows = [
        {"mention_id": "m1", "doc_id": "d1", "surface": "Siemens Corp"},
        {"mention_id": "m2", "doc_id": "d1", "surface": "Mentor Graphics Inc"},
        {"mention_id": "m3", "doc_id": "d1", "surface": "Unrelated Thing"},
    ]
    gold_rows = [{"doc_id": "d1", "subject": "Siemens", "object": "Mentor Graphics"}]
    clusters = gold_clusters_for_mentions(mention_rows, gold_rows, entity_threshold=0.85)
    assert clusters == {"m1": "d1:siemens", "m2": "d1:mentor graphics"}
    assert "m3" not in clusters


def test_predicted_clusters_from_store(tmp_store) -> None:
    tmp_store.insert_document("d1", "d1.pdf", "sha", 10, Split.TUNE)
    tmp_store.save_canonical_text("d1", "x" * 10)
    tmp_store.insert_mention("m1", "run1", "d1", "Siemens", 0, 7, entity_id="e1")
    tmp_store.insert_mention("m2", "run1", "d1", "Siemens AG", 0, 10, entity_id="e1")
    tmp_store.insert_mention("m3", "run1", "d1", "Unresolved", 0, 10, entity_id=None)
    clusters = predicted_clusters_from_store(tmp_store, "run1", {"d1"})
    assert clusters == {"m1": "e1", "m2": "e1"}


def test_evaluate_run_reports_na_not_fabricated_zero_when_nothing_recorded(tmp_store) -> None:
    """The bug this whole pass fixes: before `run_stats`/judgements existed,
    `evaluate_run`'s uncalled-kwargs defaulted to `0`, so
    `span_localization_failure_rate` silently reported a fabricated `0.0%`
    for a run that never measured it. It must report `None` (n/a) instead."""
    tmp_store.insert_document("d1", "d1.pdf", "sha", 20, Split.TUNE)
    tmp_store.save_canonical_text("d1", "Siemens acquired Mentor Graphics.")
    tmp_store.insert_mention("subj", "run1", "d1", "Siemens", 0, 7)
    tmp_store.insert_mention("obj", "run1", "d1", "Mentor Graphics", 17, 32)
    tmp_store.insert_triple(
        Triple(
            triple_id="t1", run_id="run1", doc_id="d1", chunk_id="c1",
            subject_mention_id="subj", object_mention_id="obj",
            relation=Relation.acquired, evidence_start=0, evidence_end=32,
            evidence_text="Siemens acquired Mentor Graphics", confidence=0.9,
        )
    )
    tmp_store.start_run("run1", "a1", 0, "fake", "{}")

    m = evaluate_run(tmp_store, "run1", Split.TUNE)
    assert m.span_localization_failure_rate is None
    assert m.adjudication_rate is None
    assert m.hallucination is None
    assert m.hallucination_llm is None
    assert m.entailment_kappa is None
    assert m.er_pairwise is None


def test_evaluate_run_reads_persisted_stats_and_judgements(tmp_store) -> None:
    """End-to-end wiring check: extraction/resolution stats and human+LLM
    entailment verdicts written by separate calls (as the real `extract` /
    `resolve` / `judge` CLI commands do) are picked up by `evaluate_run`
    without the caller passing anything explicitly."""
    tmp_store.insert_document("d1", "d1.pdf", "sha", 40, Split.TUNE)
    text = "Siemens acquired Mentor Graphics. Foo bar baz qux quux corge."
    tmp_store.save_canonical_text("d1", text)
    tmp_store.insert_gold_triple(
        GoldTriple(
            gold_id="g1", doc_id="d1", subject="Siemens", relation=Relation.acquired,
            object="Mentor Graphics", evidence_start=0, evidence_end=33, annotator="t",
        )
    )
    tmp_store.insert_mention("subj", "run1", "d1", "Siemens", 0, 7)
    tmp_store.insert_mention("obj", "run1", "d1", "Mentor Graphics", 17, 32)
    tmp_store.insert_triple(
        Triple(
            triple_id="t1", run_id="run1", doc_id="d1", chunk_id="c1",
            subject_mention_id="subj", object_mention_id="obj",
            relation=Relation.acquired, evidence_start=0, evidence_end=32,
            evidence_text=text[0:32], confidence=0.9,
        )
    )
    tmp_store.start_run("run1", "a1", 0, "fake", "{}")
    tmp_store.record_extraction_stats(
        "run1", n_chunks=1, n_extracted=1, n_stored=1,
        n_span_localization_failures=2, n_deduplicated=0, n_unmapped_relations=0,
    )
    tmp_store.record_resolution_stats(
        "run1", n_mentions=2, n_candidate_pairs=10, n_accepted_direct=4,
        n_rejected=2, n_deferred=4, n_adjudicated_same=1, n_entities=1,
    )
    tmp_store.insert_entailment_judgement("t1", "run1", "human", True)
    tmp_store.insert_entailment_judgement("t1", "run1", "llm", True)

    m = evaluate_run(tmp_store, "run1", Split.TUNE)

    # 2 dropped / (2 dropped + 1 emitted) = 2/3
    assert round(m.span_localization_failure_rate, 4) == round(2 / 3, 4)
    # "LLM-adjudicated pairs" = every pair sent to adjudication = n_deferred,
    # not n_adjudicated_same (§9.2 note in c2k/eval/metrics.py).
    assert m.adjudication_rate == 0.4
    assert m.hallucination == 0.0  # both judged True == "entails"
    assert m.hallucination_llm == 0.0
    assert m.entailment_kappa == 1.0  # human and llm agree on the one judged triple
    assert m.triple_loose.precision == 1.0  # exact match, no unmatchable predictions


def test_evaluate_run_ignores_predictions_from_other_splits(tmp_store) -> None:
    """Gold exists only for the scored split, so triples from another split's
    documents used to sit in the precision denominator and could never match."""
    text = "Siemens acquired Mentor Graphics."
    for doc_id, split in (("d_tune", Split.TUNE), ("d_test", Split.TEST)):
        tmp_store.insert_document(doc_id, f"{doc_id}.pdf", "sha", len(text), split)
        tmp_store.save_canonical_text(doc_id, text)
        tmp_store.insert_mention(f"{doc_id}:s", "run1", doc_id, "Siemens", 0, 7)
        tmp_store.insert_mention(f"{doc_id}:o", "run1", doc_id, "Mentor Graphics", 17, 32)
        tmp_store.insert_triple(
            Triple(
                triple_id=f"t_{doc_id}", run_id="run1", doc_id=doc_id, chunk_id="c1",
                subject_mention_id=f"{doc_id}:s", object_mention_id=f"{doc_id}:o",
                relation=Relation.acquired, evidence_start=0, evidence_end=32,
                evidence_text="Siemens acquired Mentor Graphics", confidence=0.9,
            )
        )
    tmp_store.start_run("run1", "a1", 0, "fake", "{}")

    loaded = load_predicted(tmp_store, "run1", split=Split.TUNE)
    assert [t.id for t in loaded] == ["t_d_tune"]
    assert len(load_predicted(tmp_store, "run1")) == 2
