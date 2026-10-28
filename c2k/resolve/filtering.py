"""rapidfuzz -> accept / reject / defer (§6.4).

    score < tau_lo   -> reject
    score > tau_hi   -> accept
    otherwise        -> defer to adjudication

`tau_lo = 0.72`, `tau_hi = 0.88` are defaults, tuned on the `tune` split only
(config/default.yaml).
"""

from __future__ import annotations

from enum import Enum

from rapidfuzz import fuzz

from c2k.norm import norm


class Decision(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    DEFER = "defer"


def score_pair(a: str, b: str) -> float:
    return fuzz.token_set_ratio(norm(a), norm(b)) / 100.0


def decide(score: float, *, tau_lo: float = 0.72, tau_hi: float = 0.88) -> Decision:
    if score < tau_lo:
        return Decision.REJECT
    if score > tau_hi:
        return Decision.ACCEPT
    return Decision.DEFER


def filter_pair(a: str, b: str, *, tau_lo: float = 0.72, tau_hi: float = 0.88) -> tuple[Decision, float]:
    score = score_pair(a, b)
    return decide(score, tau_lo=tau_lo, tau_hi=tau_hi), score
