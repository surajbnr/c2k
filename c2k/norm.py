"""Shared string normalisation (§6.4): lowercase, strip punctuation and legal
suffixes, collapse whitespace. Used by both `resolve/filtering.py` (mention
resolution) and `eval/match.py` (gold matching) — the same definition, so a
mention accepted as "the same entity" during resolution is judged by an
identical rule at evaluation time.
"""

from __future__ import annotations

import re
import string

_LEGAL_SUFFIXES = re.compile(r"\b(inc|ltd|gmbh|corp)\b\.?", re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_WHITESPACE = re.compile(r"\s+")


def norm(s: str) -> str:
    s = s.lower()
    s = _LEGAL_SUFFIXES.sub("", s)
    s = s.translate(_PUNCT_TABLE)
    s = _WHITESPACE.sub(" ", s).strip()
    return s
