"""Understanding a question: which relation(s) it asks about, and about what.

Keyword rules (`c2k.query.parse_query`) catch common phrasings. For anything
worded differently, a small local embedding model compares the *meaning* of
the question with the relation definitions in relations.yaml and with example
questions in config/relation_phrases.yaml. Everything runs locally: no API
key, no per-question cost. If the embedding library or model is unavailable,
`understand` falls back to the keyword rules alone.

Names in the question are replaced with "X" before comparing meanings, so
"Where is NVIDIA based?" is judged on "Where is X based?". When two relations
are about equally likely (founded-by vs authored-by, or the company-deal
relations), all of them are returned rather than a coin-flip winner.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import yaml

from c2k.config import RELATION_PHRASES_PATH
from c2k.query import _STOPWORDS, analyse_relation
from c2k.schema import RELATION_DEFS

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
MIN_SCORE = 0.60  # best relation must score at least this, else no relation
BAND = 0.03  # also return relations scoring within this of the best
MAX_RELATIONS = 3


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...
    def query_embed(self, texts: list[str]) -> np.ndarray: ...
    def passage_embed(self, texts: list[str]) -> np.ndarray: ...


class FastEmbedder:
    """The real embedder (ONNX, CPU). Imported lazily, so the rest of c2k
    never needs the library."""

    def __init__(self, model: str = EMBED_MODEL, cache_dir: str | os.PathLike | None = None):
        from fastembed import TextEmbedding

        cache = cache_dir or os.environ.get("C2K_EMBED_CACHE") or Path.home() / ".cache" / "c2k_fastembed"
        self._model = TextEmbedding(model, cache_dir=str(cache))

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array(list(self._model.embed(texts)))

    def query_embed(self, texts: list[str]) -> np.ndarray:
        return np.array(list(self._model.query_embed(texts)))

    def passage_embed(self, texts: list[str]) -> np.ndarray:
        return np.array(list(self._model.passage_embed(texts)))


def _unit(a: np.ndarray) -> np.ndarray:
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


class RelationMatcher:
    """Scores a question against every relation by meaning."""

    def __init__(self, embedder: Embedder | None = None):
        raw = yaml.safe_load(RELATION_PHRASES_PATH.read_text(encoding="utf-8"))
        examples: dict[str, list[str]] = raw["examples"]
        unknown = set(examples) - set(RELATION_DEFS)
        if unknown:
            raise ValueError(f"relation_phrases.yaml examples name unknown relation(s): {sorted(unknown)}")
        self._embedder = embedder or FastEmbedder()
        self._names = list(RELATION_DEFS)

        doc_texts, self._doc_rel = [], []
        ex_texts, self._ex_rel = [], []
        for name in self._names:
            entry = RELATION_DEFS[name]
            for text in [entry["definition"], *entry["exemplars"]]:
                doc_texts.append(text)
                self._doc_rel.append(name)
            for text in examples.get(name, []):
                ex_texts.append(text)
                self._ex_rel.append(name)
        self._docs = _unit(self._embedder.passage_embed(doc_texts))
        self._examples = _unit(self._embedder.embed(ex_texts))

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    def scores(self, question: str) -> dict[str, float]:
        """Score per relation (higher means closer in meaning): the average of
        the best match against the relation's definition and exemplars and the
        best match against its example questions."""
        q_plain = _unit(self._embedder.embed([question]))[0]
        q_query = _unit(self._embedder.query_embed([question]))[0]
        best_doc = {n: -1.0 for n in self._names}
        for sim, name in zip(self._docs @ q_query, self._doc_rel):
            best_doc[name] = max(best_doc[name], float(sim))
        best_ex = {n: -1.0 for n in self._names}
        for sim, name in zip(self._examples @ q_plain, self._ex_rel):
            best_ex[name] = max(best_ex[name], float(sim))
        return {n: 0.5 * best_doc[n] + 0.5 * best_ex[n] for n in self._names}

    def relations(self, question: str) -> list[str]:
        """The best relation, plus any nearly as good; empty if nothing is close."""
        ranked = sorted(self.scores(question).items(), key=lambda kv: -kv[1])
        if not ranked or ranked[0][1] < MIN_SCORE:
            return []
        return [name for name, score in ranked[:MAX_RELATIONS] if score >= ranked[0][1] - BAND]


_TOKEN_EDGE = "?.,!\"'()"


def _collapse(tokens: list[str]) -> str:
    out: list[str] = []
    for t in tokens:
        if t == "X" and out and out[-1] == "X":
            continue
        out.append(t)
    return " ".join(out)


def _looks_like_name(core: str, index: int) -> bool:
    return bool(core) and core.lower() not in _STOPWORDS and (
        (core[0].isupper() and index > 0)
        or bool(re.search(r"\d", core))
        or bool(re.search(r"[a-z][A-Z]", core))
        or (core.isupper() and len(core) > 1)
        or ("-" in core and any(c.isupper() for c in core))
    )


def _is_name(core: str, index: int, entity_vocab) -> bool:
    return _looks_like_name(core, index) or (bool(core) and core.lower() not in _STOPWORDS and core.lower() in entity_vocab)


def _core(token: str) -> str:
    return token.strip(_TOKEN_EDGE).replace("'s", "")


def mask_names(question: str, entity_vocab: frozenset[str] | set[str] = frozenset()) -> str:
    """Replaces names with "X": capitalised words (not the first word), words
    with digits, camelCase or hyphenated capitals, and any word in
    `entity_vocab` (lower-case words that occur in the data's entity names, so
    an all-lowercase question is masked too)."""
    out = []
    for i, tok in enumerate(question.split()):
        out.append("X" if _is_name(_core(tok), i, entity_vocab) else tok)
    return _collapse(out)


def find_names(
    question: str,
    entity_vocab: frozenset[str] | set[str] = frozenset(),
    skip_spans: list[tuple[int, int]] | None = None,
) -> list[str]:
    """The names in a question, in order, leaving out words that belong to the
    relation phrase (`skip_spans`), e.g. "charge" in "who is in charge of
    Disney"."""
    skip_spans = skip_spans or []
    names = []
    for i, m in enumerate(re.finditer(r"\S+", question)):
        if any(m.start() < end and m.end() > start for start, end in skip_spans):
            continue
        core = _core(m.group())
        if _is_name(core, i, entity_vocab):
            names.append(core)
    return names


def entity_vocabulary(names: list[str], *, max_words: int = 3) -> frozenset[str]:
    """Lower-case words from short entity names (company, method, person
    names), minus ordinary question words."""
    vocab: set[str] = set()
    for name in names:
        words = name.lower().split()
        if 1 <= len(words) <= max_words:
            vocab.update(w.strip(_TOKEN_EDGE) for w in words)
    return frozenset(w for w in vocab if w and w not in _STOPWORDS)


@dataclass(frozen=True)
class Lexicon:
    """What the data knows about names. `mask_vocab` (words from short entity
    names) decides which words are names when masking a question; `known_words`
    (every word of every entity name) decides which leftover words may form
    the entity, so filler like "situated" or "charge" is left out."""

    mask_vocab: frozenset[str] = frozenset()
    known_words: frozenset[str] = frozenset()

    @classmethod
    def from_names(cls, names: list[str]) -> "Lexicon":
        words = {w.strip(_TOKEN_EDGE) for n in names for w in n.lower().split()}
        known = frozenset(w for w in words if len(w) > 1 and w not in _STOPWORDS)
        return cls(entity_vocabulary(names), known)


@dataclass(frozen=True)
class Understanding:
    relations: tuple[str, ...]
    entity: str | None
    how: str  # "keywords", "meaning", "both" or "none"
    scores: dict[str, float] = field(default_factory=dict)
    ignored: tuple[str, ...] = ()  # leftover words that matched nothing in the data
    widened: bool = False  # the search was widened to every kind of fact (see web/search.py ask)


def _leftover_words(question: str, skip_spans: list[tuple[int, int]]) -> list[str]:
    words = []
    for m in re.finditer(r"\S+", question):
        if any(m.start() < end and m.end() > start for start, end in skip_spans):
            continue
        core = _core(m.group()).strip(".,'\"-/")
        if core and core.lower() not in _STOPWORDS:
            words.append(core)
    return words


def _pick_entity(question: str, spans: list[tuple[int, int]], lexicon: Lexicon) -> tuple[str | None, tuple[str, ...]]:
    names = find_names(question, lexicon.mask_vocab, spans)
    if not lexicon.known_words:  # no data to check against (e.g. the command line)
        leftover = names or _leftover_words(question, spans)
        return (" ".join(leftover) or None), ()
    chosen: list[str] = []
    ignored: list[str] = []
    name_set = {n.lower() for n in names}
    for word in _leftover_words(question, spans):
        if word.lower() in name_set or word.lower() in lexicon.known_words:
            chosen.append(word)
        else:
            ignored.append(word)
    return (" ".join(chosen) or None), tuple(ignored)


def understand(
    question: str,
    matcher: RelationMatcher | None = None,
    lexicon: Lexicon | None = None,
) -> Understanding:
    lexicon = lexicon or Lexicon()
    rule_relation, spans = analyse_relation(question)
    rule = rule_relation.value if rule_relation else None
    entity, ignored = _pick_entity(question, spans, lexicon)
    scores: dict[str, float] = {}
    meaning: list[str] = []
    if matcher is not None:
        masked = mask_names(question, lexicon.mask_vocab)
        scores = matcher.scores(masked)
        meaning = matcher.relations(masked)

    if rule and (not meaning or rule in meaning):
        return Understanding((rule,), entity, "both" if meaning else "keywords", scores, ignored)
    if rule:
        return Understanding(tuple(meaning) + (rule,), entity, "both", scores, ignored)
    if meaning:
        return Understanding(tuple(meaning), entity, "meaning", scores, ignored)
    return Understanding((), entity, "none", scores, ignored)
