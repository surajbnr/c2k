"""Embeddings + FAISS -> candidate pairs (§6.4).

Each mention is embedded as `f"{surface} | {context}"` where context is
+-100 characters from the canonical stream. `sentence-transformers` and
`faiss` are imported lazily so importing this module doesn't require either
to be installed unless blocking actually runs.
"""

from __future__ import annotations

from dataclasses import dataclass

CONTEXT_WINDOW = 100


@dataclass(frozen=True)
class MentionRecord:
    mention_id: str
    doc_id: str
    surface: str
    char_start: int
    char_end: int


def mention_embedding_text(mention: MentionRecord, canonical_text: str) -> str:
    lo = max(0, mention.char_start - CONTEXT_WINDOW)
    hi = min(len(canonical_text), mention.char_end + CONTEXT_WINDOW)
    context = canonical_text[lo:hi]
    return f"{mention.surface} | {context}"


def embed_mentions(
    mentions: list[MentionRecord], canonical_by_doc: dict[str, str], *, model_name: str
):
    """Returns an (n, d) float32 array of L2-normalised embeddings, one row
    per mention, in the same order as `mentions`."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    texts = [mention_embedding_text(m, canonical_by_doc[m.doc_id]) for m in mentions]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embeddings, dtype="float32")


def block_candidates(embeddings, *, top_k: int = 20) -> list[tuple[int, int]]:
    """`IndexFlatIP` over L2-normalised embeddings, top-k=20 per mention.
    Returns deduplicated, order-independent (i, j) index pairs with i < j."""
    import faiss

    n = embeddings.shape[0]
    if n < 2:
        return []
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    k = min(top_k + 1, n)  # +1 because a mention is always its own nearest neighbour
    _, neighbors = index.search(embeddings, k)

    pairs: set[tuple[int, int]] = set()
    for i, row in enumerate(neighbors):
        for j in row:
            j = int(j)
            if j == -1 or j == i:
                continue
            pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)
