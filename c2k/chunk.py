"""Canonical text -> overlapping chunks (§6.2).

900 tokens, 150-token overlap, `cl100k_base` for counting, boundaries snapped
to the nearest sentence break within +-50 tokens.

Chunk boundaries are laid out on a token grid (independent of snapping) so the
stride/overlap arithmetic stays exact; only the *stored* char boundary of each
interior cut is nudged to the nearest sentence end within the token-derived
character window. The final chunk's end is never snapped (nothing to snap to).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken

from c2k.store import Store

_ENCODING_NAME = "cl100k_base"
_SENTENCE_END = re.compile(r'[.!?]["\')\]]?(?=\s|$)')


@dataclass(frozen=True)
class ChunkSpan:
    chunk_id: str
    doc_id: str
    idx: int
    char_start: int
    char_end: int
    text: str


def _char_offset(enc: tiktoken.Encoding, token_ids: list[int], token_idx: int) -> int:
    if token_idx <= 0:
        return 0
    if token_idx >= len(token_ids):
        return len(enc.decode(token_ids))
    return len(enc.decode(token_ids[:token_idx]))


def _snap_to_sentence(text: str, char_pos: int, lo: int, hi: int) -> int:
    lo = max(0, lo)
    hi = min(len(text), hi)
    candidates = [m.end() for m in _SENTENCE_END.finditer(text, lo, hi)]
    if not candidates:
        return char_pos
    return min(candidates, key=lambda c: abs(c - char_pos))


def chunk_text(
    doc_id: str,
    text: str,
    *,
    tokens: int = 900,
    overlap: int = 150,
    snap_window: int = 50,
) -> list[ChunkSpan]:
    if overlap >= tokens:
        raise ValueError("overlap must be smaller than tokens")
    enc = tiktoken.get_encoding(_ENCODING_NAME)
    token_ids = enc.encode(text, disallowed_special=())
    n = len(token_ids)
    if n == 0:
        return []

    stride = tokens - overlap
    chunks: list[ChunkSpan] = []
    idx = 0
    start_tok = 0
    while start_tok < n:
        end_tok = min(start_tok + tokens, n)
        char_start = _char_offset(enc, token_ids, start_tok)
        char_end = _char_offset(enc, token_ids, end_tok)

        if end_tok < n:
            lo_tok = max(0, end_tok - snap_window)
            hi_tok = min(n, end_tok + snap_window)
            lo_char = _char_offset(enc, token_ids, lo_tok)
            hi_char = _char_offset(enc, token_ids, hi_tok)
            char_end = _snap_to_sentence(text, char_end, lo_char, hi_char)

        chunks.append(
            ChunkSpan(
                chunk_id=f"{doc_id}:{idx}",
                doc_id=doc_id,
                idx=idx,
                char_start=char_start,
                char_end=char_end,
                text=text[char_start:char_end],
            )
        )

        if end_tok >= n:
            break
        idx += 1
        start_tok += stride

    return chunks


def chunk_document(
    store: Store,
    doc_id: str,
    *,
    tokens: int = 900,
    overlap: int = 150,
    snap_window: int = 50,
) -> list[ChunkSpan]:
    text = store.load_canonical_text(doc_id)
    chunks = chunk_text(doc_id, text, tokens=tokens, overlap=overlap, snap_window=snap_window)
    for c in chunks:
        assert text[c.char_start : c.char_end] == c.text
        store.insert_chunk(c.chunk_id, c.doc_id, c.idx, c.char_start, c.char_end, c.text)
    return chunks
