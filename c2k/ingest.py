"""PDF -> canonical text + offset index (§6.1).

Builds a single UTF-8 canonical text stream per document: words from
`page.get_text("words")` joined with single spaces, pages joined with
`"\\n\\n"`. Every downstream offset indexes into this stream. Alongside it we
persist a `(char_start, char_end) -> (page, bbox)` index so a span can be
traced back to its location on the page image.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from c2k.config import load_splits
from c2k.store import Split, Store


def compute_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def doc_id_for(pdf_path: Path) -> str:
    return pdf_path.stem


def build_canonical_text(pdf_path: Path) -> tuple[str, list[dict]]:
    """Returns (canonical_text, spans) where spans is a list of
    {"start", "end", "page", "bbox"} dicts, one per word, in the order they
    appear in the canonical text."""
    import pymupdf  # imported lazily so the module can be read/tested without it installed

    doc = pymupdf.open(pdf_path)
    try:
        pieces: list[str] = []
        spans: list[dict] = []
        pos = 0
        n_pages = len(doc)
        for page_no in range(n_pages):
            page = doc[page_no]
            words = page.get_text("words")  # (x0,y0,x1,y1,word,block,line,word_no)
            for i, w in enumerate(words):
                x0, y0, x1, y1, word = w[0], w[1], w[2], w[3], w[4]
                if i > 0:
                    pieces.append(" ")
                    pos += 1
                start = pos
                pieces.append(word)
                pos += len(word)
                spans.append({"start": start, "end": pos, "page": page_no, "bbox": [x0, y0, x1, y1]})
            if page_no != n_pages - 1:
                pieces.append("\n\n")
                pos += 2
        return "".join(pieces), spans
    finally:
        doc.close()


def ingest_pdf(store: Store, pdf_path: Path, splits: dict[str, str]) -> str:
    doc_id = doc_id_for(pdf_path)
    text, spans = build_canonical_text(pdf_path)
    store.save_canonical_text(doc_id, text)
    (store.canonical_dir / f"{doc_id}.spans.json").write_text(json.dumps(spans))
    sha = compute_sha256(pdf_path)
    split = splits.get(doc_id, Split.TRAIN_UNUSED.value)
    store.insert_document(doc_id, str(pdf_path), sha, len(text), split)
    return doc_id


def ingest_directory(
    store: Store, input_dir: Path | str, splits: dict[str, str] | None = None
) -> list[str]:
    resolved_splits = load_splits() if splits is None else splits
    doc_ids = []
    for pdf_path in sorted(Path(input_dir).glob("*.pdf")):
        doc_ids.append(ingest_pdf(store, pdf_path, resolved_splits))
    return doc_ids
