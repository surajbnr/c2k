"""§9.1 / §15: Hungarian alignment must be one-to-one and independent of row
order — a greedy matcher would fail the shuffle test below."""

from __future__ import annotations

import random

from c2k.eval.match import EvalTriple, match
from c2k.schema import Relation


def _triples() -> tuple[list[EvalTriple], list[EvalTriple]]:
    predicted = [
        EvalTriple("p0", "Siemens", Relation.acquired, "Mentor Graphics"),
        EvalTriple("p1", "Siemens AG", Relation.acquired, "Mentor Graphics Inc"),  # near-dup of p0's gold
        EvalTriple("p2", "Acme Corp", Relation.headquartered_in, "Springfield"),
        EvalTriple("p3", "Totally Unrelated", Relation.founded_by, "Nobody"),
    ]
    gold = [
        EvalTriple("g0", "Siemens", Relation.acquired, "Mentor Graphics"),
        EvalTriple("g1", "Acme Corporation", Relation.headquartered_in, "Springfield"),
    ]
    return predicted, gold


def test_matching_is_one_to_one() -> None:
    predicted, gold = _triples()
    matches = match(predicted, gold, entity_threshold=0.85)
    pred_idxs = [p for p, _ in matches]
    gold_idxs = [g for _, g in matches]
    assert len(pred_idxs) == len(set(pred_idxs))
    assert len(gold_idxs) == len(set(gold_idxs))
    # p0 and p1 are both eligible for g0; only one may be taken.
    assert sum(1 for p, g in matches if g == 0) <= 1


def test_matching_is_order_independent() -> None:
    predicted, gold = _triples()
    baseline = match(predicted, gold, entity_threshold=0.85)
    baseline_ids = {(predicted[p].id, gold[g].id) for p, g in baseline}

    rng = random.Random(42)
    for _ in range(20):
        shuffled_pred = predicted[:]
        shuffled_gold = gold[:]
        rng.shuffle(shuffled_pred)
        rng.shuffle(shuffled_gold)
        result = match(shuffled_pred, shuffled_gold, entity_threshold=0.85)
        result_ids = {(shuffled_pred[p].id, shuffled_gold[g].id) for p, g in result}
        assert result_ids == baseline_ids


def test_no_match_across_different_relations() -> None:
    predicted = [EvalTriple("p0", "X", Relation.acquired, "Y")]
    gold = [EvalTriple("g0", "X", Relation.founded_by, "Y")]
    assert match(predicted, gold) == []


def test_span_strict_variant_requires_iou() -> None:
    predicted = [EvalTriple("p0", "X", Relation.acquired, "Y", evidence_start=0, evidence_end=10)]
    gold_far = [EvalTriple("g0", "X", Relation.acquired, "Y", evidence_start=1000, evidence_end=1010)]
    gold_close = [EvalTriple("g0", "X", Relation.acquired, "Y", evidence_start=0, evidence_end=10)]

    assert match(predicted, gold_far, span_iou_threshold=0.5) == []
    assert match(predicted, gold_close, span_iou_threshold=0.5) == [(0, 0)]
    # loose variant ignores spans entirely
    assert match(predicted, gold_far) == [(0, 0)]
