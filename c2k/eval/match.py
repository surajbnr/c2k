"""Gold <-> predicted alignment (§9.1).

Matching is relation-exact plus fuzzy subject/object similarity, solved as a
one-to-one assignment with `scipy.optimize.linear_sum_assignment` so results
are deterministic and independent of row order. There is deliberately no
greedy fallback anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

from c2k.norm import norm
from c2k.schema import Relation

__all__ = ["norm", "sim", "iou", "EvalTriple", "match"]

# Sentinel cost for ineligible pairs in the assignment matrix. scipy's
# Hungarian solver assigns every row (or column) in the smaller dimension, so
# ineligible pairs get a very large cost and are filtered out of the result
# afterwards by their score, not by the assignment itself.
_INELIGIBLE_COST = 1e6


def sim(a: str, b: str) -> float:
    return fuzz.token_sort_ratio(a, b) / 100.0


def iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class EvalTriple:
    """A relation-labelled fact reduced to what the matcher needs: surface
    strings and (optionally) an evidence span. Used for both predicted
    triples (joined from the DB) and gold triples."""

    id: str
    subject: str
    relation: Relation
    object: str
    evidence_start: int | None = None
    evidence_end: int | None = None


def _eligible(
    p: EvalTriple,
    g: EvalTriple,
    *,
    entity_threshold: float,
    span_iou_threshold: float | None,
) -> tuple[bool, float]:
    if p.relation != g.relation:
        return False, 0.0
    s_subj = sim(norm(p.subject), norm(g.subject))
    s_obj = sim(norm(p.object), norm(g.object))
    if s_subj < entity_threshold or s_obj < entity_threshold:
        return False, 0.0
    if span_iou_threshold is not None:
        if None in (p.evidence_start, p.evidence_end, g.evidence_start, g.evidence_end):
            return False, 0.0
        span_score = iou(p.evidence_start, p.evidence_end, g.evidence_start, g.evidence_end)
        if span_score < span_iou_threshold:
            return False, 0.0
    return True, (s_subj + s_obj) / 2.0


def match(
    predicted: list[EvalTriple],
    gold: list[EvalTriple],
    *,
    entity_threshold: float = 0.85,
    span_iou_threshold: float | None = None,
) -> list[tuple[int, int]]:
    """Returns a list of (predicted_idx, gold_idx) one-to-one matches.

    Pass `span_iou_threshold` for the span-strict variant (§9.1); leave it
    `None` for the headline loose variant.
    """
    if not predicted or not gold:
        return []

    scores = np.full((len(predicted), len(gold)), -1.0)
    for i, p in enumerate(predicted):
        for j, g in enumerate(gold):
            eligible, score = _eligible(
                p, g, entity_threshold=entity_threshold, span_iou_threshold=span_iou_threshold
            )
            if eligible:
                scores[i, j] = score

    cost = np.where(scores >= 0, -scores, _INELIGIBLE_COST)
    row_ind, col_ind = linear_sum_assignment(cost)
    return [(int(r), int(c)) for r, c in zip(row_ind, col_ind) if scores[r, c] >= 0]
