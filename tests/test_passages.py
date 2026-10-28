"""Passage search (web/passages.py): quoting the closest passages of a document."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from web import passages as P

VOCAB = ["price", "billion", "acquire", "weather", "toronto", "city", "signed", "agreement", "hugging", "face", "joke", "recipe"]


class BagOfWords:
    """Tiny stand-in embedder: counts of a fixed vocabulary, so cosine similarity behaves sensibly."""

    def _vec(self, texts):
        rows = []
        for t in texts:
            words = [w.strip(".,?!$").lower() for w in t.split()]
            rows.append([float(sum(w.startswith(v) for w in words)) for v in VOCAB] + [0.01])
        return np.array(rows)

    embed = query_embed = passage_embed = lambda self, texts: self._vec(texts)


DOC = (
    "NVIDIA signed a definitive agreement to acquire Hugging Face. "
    "The transaction includes an approximately $11.9 billion purchase price. "
    "Separately, the weather in Toronto was mild during the study period. " * 3
)


def test_windows_cover_the_text_in_order_and_stay_short() -> None:
    windows = P.build_windows(DOC)
    assert windows and all(len(w) <= P.MAX_WINDOW_CHARS + 200 for w in windows)
    assert " ".join(windows).split() == " ".join(DOC.split()).split()


def test_a_question_finds_the_passage_that_answers_it() -> None:
    index = P.build_index(DOC, BagOfWords())
    found = P.search({"deal.pdf": index}, "what is the purchase price in billion", BagOfWords(), min_score=0.3)
    assert found and "11.9 billion" in found[0].text and found[0].doc_name == "deal.pdf"


def test_an_unrelated_question_finds_nothing() -> None:
    index = P.build_index(DOC, BagOfWords())
    assert P.search({"deal.pdf": index}, "tell me a joke and a recipe", BagOfWords(), min_score=0.5) == []


def test_results_are_best_first_limited_and_span_documents() -> None:
    a = P.build_index("The price was one billion. Unrelated words here.", BagOfWords())
    b = P.build_index("The weather in Toronto.", BagOfWords())
    found = P.search({"a.pdf": a, "b.pdf": b}, "price billion", BagOfWords(), min_score=0.1, k=1)
    assert len(found) == 1 and found[0].doc_name == "a.pdf"


def test_empty_inputs_are_safe() -> None:
    assert P.search({}, "anything", BagOfWords()) == []
    assert P.search({"x": P.build_index("", BagOfWords())}, "anything", BagOfWords()) == []
    assert P.search({"x": P.build_index(DOC, BagOfWords())}, "   ", BagOfWords()) == []


def _model_is_cached() -> bool:
    cache = Path(os.environ.get("C2K_EMBED_CACHE") or Path.home() / ".cache" / "c2k_fastembed")
    return cache.exists() and any(cache.iterdir())


@pytest.mark.skipif(not _model_is_cached(), reason="embedding model not downloaded")
def test_with_the_real_model_relevant_questions_hit_and_irrelevant_ones_do_not() -> None:
    from c2k.understand import FastEmbedder

    emb = FastEmbedder()
    text = (
        "NVIDIA entered into a definitive agreement on September 2, 2026 to acquire Hugging Face, Inc. "
        "The transaction includes an approximately $11.9 billion purchase price payable to stockholders. "
        "Hugging Face operates a platform and community for developing, sharing and deploying open-source models. "
        "The closing is subject to customary regulatory approvals and other conditions."
    )
    index = P.build_index(text, emb)
    price = P.search({"d.pdf": index}, "how much did NVIDIA pay for Hugging Face", emb)
    assert price and "11.9 billion" in price[0].text
    for q in ("what's the weather like", "give me a pancake recipe", "how tall is mount everest"):
        assert P.search({"d.pdf": index}, q, emb) == [], q
