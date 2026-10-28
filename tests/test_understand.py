"""Question understanding (c2k/understand.py): the plumbing is tested with a
fake embedder so it runs anywhere; accuracy on held-out questions is checked
with the real model when it has been downloaded."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from c2k.understand import (
    BAND,
    MIN_SCORE,
    Lexicon,
    RelationMatcher,
    Understanding,
    entity_vocabulary,
    find_names,
    mask_names,
    understand,
)

DATA = Path(__file__).resolve().parent / "data"


class FakeEmbedder:
    """Deterministic stand-in: a text maps to a repeatable pseudo-random unit vector."""

    def _vec(self, texts):
        out = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            out.append(rng.normal(size=16))
        return np.array(out)

    embed = query_embed = passage_embed = lambda self, texts: self._vec(texts)


class FakeMatcher:
    def __init__(self, relations):
        self._relations = relations

    def scores(self, question):
        return {}

    def relations(self, question):
        return list(self._relations)


# ---------------------------------------------------------------- masking / names


def test_mask_names_covers_capitalised_codes_and_vocabulary_words() -> None:
    assert mask_names("Where is Apple headquartered?") == "Where is X headquartered?"
    assert mask_names("what does GPT-4o beat") == "what does X beat"
    assert mask_names("which models does PT-TTT surpass") == "which models does X surpass"
    assert mask_names("where is apple located", frozenset({"apple"})) == "where is X located"
    assert mask_names("who is Julita Vassileva from") == "who is X from"


def test_find_names_skips_words_that_belong_to_the_relation_phrase() -> None:
    names = find_names("who is in charge of Disney", frozenset(), skip_spans=[(10, 16)])
    assert names == ["Disney"]


def test_entity_vocabulary_uses_short_names_and_drops_question_words() -> None:
    vocab = entity_vocabulary(["Apple Inc.", "Walt Disney", "The Walt Disney Company", "a b c d e", "Hugging Face"])
    assert {"apple", "hugging", "face", "walt", "disney"} <= vocab
    assert "the" not in vocab and "company" not in vocab
    assert "b" not in vocab and "e" not in vocab  # names longer than three words are not used


# ---------------------------------------------------------------- understand()


def test_filler_words_not_in_the_data_are_ignored_and_reported() -> None:
    lex = Lexicon.from_names(["The Walt Disney Company", "Disney", "Apple Inc."])
    u = understand("where is disney situated", None, lex)
    assert u.entity == "disney" and u.ignored == ("situated",)


def test_without_a_lexicon_leftover_words_form_the_entity() -> None:
    assert understand("where is apple located?").entity == "apple"


def test_keywords_alone_when_there_is_no_matcher() -> None:
    u = understand("who bought Hugging Face")
    assert u.how == "keywords" and len(u.relations) == 1


def test_meaning_alone_when_no_keyword_matches() -> None:
    u = understand("zorbulate the quux", FakeMatcher(["acquired"]))
    assert u.how == "meaning" and u.relations == ("acquired",)


def test_keyword_confirmed_by_meaning_is_not_widened() -> None:
    kw = understand("who bought Hugging Face").relations[0]
    u = understand("who bought Hugging Face", FakeMatcher([kw, "ceo_of"]))
    assert u.how == "both" and u.relations == (kw,)


def test_keyword_that_meaning_disagrees_with_is_kept_alongside() -> None:
    kw = understand("who bought Hugging Face").relations[0]
    other = "ceo_of" if kw != "ceo_of" else "acquired"
    u = understand("who bought Hugging Face", FakeMatcher([other]))
    assert u.relations == (other, kw)


def test_nothing_understood() -> None:
    assert understand("hello", FakeMatcher([])) == Understanding((), "hello", "none", {}, ())


# ---------------------------------------------------------------- matcher plumbing


def test_matcher_picks_best_and_near_ties_only(monkeypatch) -> None:
    m = RelationMatcher(FakeEmbedder())
    names = list(m.scores("x"))
    top, near, far = names[0], names[1], names[2]
    monkeypatch.setattr(m, "scores", lambda q: {top: 0.80, near: 0.80 - BAND / 2, far: 0.80 - 2 * BAND, **{n: 0.1 for n in names[3:]}})
    assert m.relations("q") == [top, near]


def test_matcher_returns_nothing_below_the_minimum_score(monkeypatch) -> None:
    m = RelationMatcher(FakeEmbedder())
    names = list(m.scores("x"))
    monkeypatch.setattr(m, "scores", lambda q: {n: MIN_SCORE - 0.01 for n in names})
    assert m.relations("q") == []


def test_matcher_returns_at_most_three_relations(monkeypatch) -> None:
    m = RelationMatcher(FakeEmbedder())
    names = list(m.scores("x"))
    monkeypatch.setattr(m, "scores", lambda q: {n: 0.9 for n in names})
    assert len(m.relations("q")) == 3


def test_example_file_naming_an_unknown_relation_is_rejected(tmp_path, monkeypatch) -> None:
    import c2k.understand as u

    bad = tmp_path / "p.yaml"
    bad.write_text("phrases: {}\nwhere_is: x\nexamples:\n  not_a_relation: ['q?']\n")
    monkeypatch.setattr(u, "RELATION_PHRASES_PATH", bad)
    with pytest.raises(ValueError, match="unknown relation"):
        u.RelationMatcher(FakeEmbedder())


# ---------------------------------------------------------------- accuracy (real model)


def _model_is_cached() -> bool:
    cache = Path(os.environ.get("C2K_EMBED_CACHE") or Path.home() / ".cache" / "c2k_fastembed")
    return cache.exists() and any(cache.iterdir())


needs_model = pytest.mark.skipif(not _model_is_cached(), reason="embedding model not downloaded")


@pytest.fixture(scope="module")
def real_matcher():
    return RelationMatcher()


@pytest.fixture(scope="module")
def site_lexicon():
    """The site's own lexicon, built from the exported data (as in production)."""
    bundle = DATA.parent.parent / "web" / "demo_data" / "demo.sqlite"
    if not bundle.exists():
        pytest.skip("run scripts/export_demo.py first")
    from web import search as S

    return S.lexicon(S.connect(bundle))


def _score(matcher, lexicon, path: Path, lower: bool) -> tuple[float, float]:
    items = yaml.safe_load(path.read_text())
    real = [i for i in items if i["relation"]]
    off = [i for i in items if not i["relation"]]
    f = (lambda q: q.lower()) if lower else (lambda q: q)
    included = sum(i["relation"] in understand(f(i["q"]), matcher, lexicon).relations for i in real)
    ignored = sum(not understand(f(i["q"]), matcher, lexicon).relations for i in off)
    return included / len(real), ignored / len(off)


@needs_model
@pytest.mark.parametrize("file", ["query_eval.yaml", "query_eval_fresh.yaml"])
@pytest.mark.parametrize("lower", [False, True])
def test_unseen_wordings_reach_the_right_relation(real_matcher, site_lexicon, file, lower) -> None:
    included, ignored = _score(real_matcher, site_lexicon, DATA / file, lower)
    assert included >= 0.85, f"{file} lower={lower}: true relation included only {included:.0%}"
    assert ignored >= 0.8, f"{file} lower={lower}: off-topic ignored only {ignored:.0%}"
