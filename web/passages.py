"""Finds the passages of an uploaded document closest in meaning to a
question, for the questions that fact extraction cannot answer (a price, a
date, anything outside the 14 kinds of fact). It only quotes the document:
no model writes anything here, so nothing can be invented."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

MAX_WINDOW_CHARS = 500
MIN_SCORE = 0.58  # relevant questions scored 0.63 and up, irrelevant ones 0.54 and below, on the samples measured
MAX_PASSAGES = 3


def build_windows(text: str, max_chars: int = MAX_WINDOW_CHARS) -> list[str]:
    """Consecutive sentences grouped into passages of roughly `max_chars`."""
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    windows: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_chars:
            windows.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        windows.append(current)
    return windows


@dataclass(frozen=True)
class PassageIndex:
    windows: list[str]
    vectors: np.ndarray  # one unit vector per window


def _unit(a: np.ndarray) -> np.ndarray:
    return a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12, None)


def build_index(text: str, embedder) -> PassageIndex:
    windows = build_windows(text)
    return PassageIndex(windows, _unit(embedder.passage_embed(windows)) if windows else np.zeros((0, 1)))


@dataclass(frozen=True)
class Passage:
    doc_name: str
    text: str
    score: float


def search(
    indexes: dict[str, PassageIndex],
    question: str,
    embedder,
    *,
    k: int = MAX_PASSAGES,
    min_score: float = MIN_SCORE,
) -> list[Passage]:
    """The best passages across the given documents (`indexes` maps a
    document's name to its index), best first, only those scoring at least
    `min_score`."""
    question = question.strip()
    if not question or not indexes:
        return []
    q = _unit(embedder.query_embed([question]))[0]
    found: list[Passage] = []
    for name, index in indexes.items():
        if not len(index.windows):
            continue
        for sim, window in zip(index.vectors @ q, index.windows):
            if sim >= min_score:
                found.append(Passage(name, window, float(sim)))
    found.sort(key=lambda p: -p.score)
    return found[:k]
