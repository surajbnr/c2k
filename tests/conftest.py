from __future__ import annotations

import random
from pathlib import Path

import pytest

from c2k.store import Store

_WORD_POOL = [
    "siemens", "acquired", "mentor", "graphics", "munich", "germany",
    "the", "board", "approved", "merger", "quarterly", "revenue", "grew",
    "researchers", "published", "results", "on", "benchmark", "dataset",
    "model", "outperforms", "baseline", "funded", "by", "grant", "agency",
    "company", "headquartered", "in", "california", "founded", "startup",
]


def make_synthetic_pdf(path: Path, *, n_pages: int = 2, lines_per_page: int = 4,
                        words_per_line: int = 8, seed: int = 0) -> list[list[str]]:
    """Writes a small PDF with known word content and returns the word list
    per line (page-major order), so tests can compute the expected canonical
    text independent of the ingest module under test."""
    import pymupdf

    rng = random.Random(seed)
    doc = pymupdf.open()
    lines_by_page: list[list[str]] = []
    for _ in range(n_pages):
        page = doc.new_page()
        page_lines = []
        y = 72
        for _ in range(lines_per_page):
            words = [rng.choice(_WORD_POOL) for _ in range(words_per_line)]
            page.insert_text((72, y), " ".join(words), fontsize=11)
            page_lines.append(words)
            y += 20
        lines_by_page.append(page_lines)
    doc.save(path)
    doc.close()
    return lines_by_page


@pytest.fixture
def tmp_store(tmp_path: Path) -> Store:
    store = Store(tmp_path / "c2k.sqlite", canonical_dir=tmp_path / "canonical")
    store.initialize()
    yield store
    store.close()
