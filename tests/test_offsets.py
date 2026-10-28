"""§6.1 / §15: canonical text slices must appear verbatim in the source PDF.

Docling was rejected specifically because its markdown export reflows
characters, breaking this property. This test is the guardrail: for 10
synthetic PDFs, 50 random slices of the canonical text each must appear
(modulo whitespace normalisation, since PyMuPDF's own "text" mode does not
promise identical spacing to the word-joined canonical stream) in the PDF's
raw extracted text.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import pytest

from c2k.ingest import build_canonical_text
from tests.conftest import make_synthetic_pdf

_WS = re.compile(r"\s+")


def _normalize(s: str) -> str:
    return _WS.sub(" ", s).strip()


@pytest.mark.parametrize("doc_idx", range(10))
def test_canonical_slices_appear_in_source(tmp_path: Path, doc_idx: int) -> None:
    import pymupdf

    pdf_path = tmp_path / f"doc_{doc_idx}.pdf"
    make_synthetic_pdf(pdf_path, seed=doc_idx)

    canonical_text, spans = build_canonical_text(pdf_path)
    assert canonical_text, "canonical text must not be empty"

    raw_doc = pymupdf.open(pdf_path)
    raw_text = _normalize("\n".join(page.get_text("text") for page in raw_doc))
    raw_doc.close()

    rng = random.Random(1000 + doc_idx)
    n = len(canonical_text)
    checked = 0
    for _ in range(50):
        if n < 5:
            break
        start = rng.randint(0, n - 5)
        length = rng.randint(3, min(20, n - start))
        slice_ = _normalize(canonical_text[start : start + length])
        if not slice_:
            continue
        assert slice_ in raw_text, f"canonical slice {slice_!r} not found in raw PDF text"
        checked += 1
    assert checked > 0


def test_span_index_covers_every_word(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    lines_by_page = make_synthetic_pdf(pdf_path, n_pages=1, lines_per_page=2, words_per_line=5)
    canonical_text, spans = build_canonical_text(pdf_path)

    expected_words = [w for line in lines_by_page[0] for w in line]
    assert len(spans) == len(expected_words)
    for span, word in zip(spans, expected_words):
        assert canonical_text[span["start"] : span["end"]] == word
        assert span["page"] == 0
        assert len(span["bbox"]) == 4
