"""NetworkX traversal + span verification (§6.6).

Loads triples into a `MultiDiGraph`; every returned fact carries
`(evidence_text, doc_id, char_start, char_end)`. Before returning, I1 is
re-verified against the canonical stream — a fact that fails verification is
dropped and logged, never returned with a warning.

Query parsing is a small heuristic, not an LLM call: this keeps `query.py`
free of any model dependency (nothing here needs schema-constrained
decoding), matching §2's "a notebook is the interface" spirit — a simple,
inspectable rule beats another moving part for a keyword-shaped query like
"which companies acquired X".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import networkx as nx
import yaml
from rapidfuzz import fuzz

from c2k.config import RELATION_PHRASES_PATH
from c2k.norm import norm
from c2k.schema import Relation
from c2k.store import Store


@dataclass(frozen=True)
class QueryResult:
    subject: str
    relation: str
    object: str
    doc_id: str
    evidence_text: str
    char_start: int
    char_end: int
    triple_id: str = ""


def build_graph(store: Store, run_id: str) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for row in store.list_triples_with_surface(run_id):
        graph.add_edge(
            row["subject_surface"],
            row["object_surface"],
            key=row["triple_id"],
            relation=row["relation"],
            doc_id=row["doc_id"],
            evidence_start=row["evidence_start"],
            evidence_end=row["evidence_end"],
            evidence_text=row["evidence_text"],
        )
    return graph


def _load_relation_phrases() -> tuple[dict[str, list[str]], str]:
    """Everyday phrasings per relation, from config/relation_phrases.yaml (the
    relation names live in YAML, not in code: invariant I3). Every key is
    checked against the ontology so the two files cannot drift apart."""
    raw = yaml.safe_load(RELATION_PHRASES_PATH.read_text(encoding="utf-8"))
    known = {r.value for r in Relation}
    unknown = (set(raw["phrases"]) | {raw["where_is"]}) - known
    if unknown:
        raise ValueError(f"relation_phrases.yaml names unknown relation(s): {sorted(unknown)}")
    return raw["phrases"], raw["where_is"]


RELATION_PHRASES, WHERE_IS_RELATION = _load_relation_phrases()
_WHERE_IS = re.compile(r"\bwhere(?:'s| is| are)\b")
_TRIGGER_RX = [
    (name, re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(pats) + r")(?![A-Za-z0-9])"))
    for name, pats in RELATION_PHRASES.items()
]

# Words that carry no entity name. Leftover words in the question are taken as
# the entity, in any letter case.
_STOPWORDS = set(
    """who whom whose what which where when why how is are was were am be been being do does did has have had can could will
    would shall should may might the a an of in on at by for from to with without about into over under between and or but not
    as than that this these those there here it its me my i you your we our they them their he she him her his tell show give
    find list get see know say all any some every each more most many much other others please company companies firm firms
    organization organisation organizations organisations people person thing things fact facts information info detail details
    paper papers study studies work research model models""".split()
)
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.&\'\-+/_]*")


def analyse_relation(text: str) -> tuple[Relation | None, list[tuple[int, int]]]:
    """Finds the relation a question asks about and the character spans of the
    words that signalled it (so they can be kept out of the entity name)."""
    lowered = text.lower()
    matches: list[tuple[int, int, str]] = []  # (length, -start, relation)
    spans: list[tuple[int, int]] = []
    for r in Relation:
        for form in {r.value, r.value.replace("_", " ")}:
            for m in re.finditer(rf"\b{re.escape(form)}\b", lowered):
                matches.append((m.end() - m.start(), -m.start(), r.value))
                spans.append((m.start(), m.end()))
    for name, rx in _TRIGGER_RX:
        for m in rx.finditer(lowered):
            matches.append((m.end() - m.start(), -m.start(), name))
            spans.append((m.start(), m.end()))

    relation = Relation(max(matches)[2]) if matches else None
    if relation is None and _WHERE_IS.search(lowered):
        relation = Relation(WHERE_IS_RELATION)
    if len(lowered) != len(text):
        spans = []
    return relation, spans


def parse_query(text: str) -> tuple[Relation | None, str | None]:
    """Turns a question into (relation, entity). The relation is found from
    the relation's own name or a common paraphrase (see
    config/relation_phrases.yaml); the entity is whatever is left once
    relation words and filler are removed, so lowercase names work. No model
    call: a simple, inspectable rule."""
    relation, spans = analyse_relation(text)
    blanked = list(text)
    for start, end in spans:
        blanked[start:end] = " " * (end - start)
    tokens = [t.rstrip(".,'\"-/") for t in _TOKEN.findall("".join(blanked))]
    tokens = [t[:-2] if t.lower().endswith("'s") else t for t in tokens]
    entity_tokens = [t for t in tokens if t and t.lower() not in _STOPWORDS]
    return relation, (" ".join(entity_tokens) if entity_tokens else None)


def names_match(query: str, name: str, *, threshold: float = 85.0) -> bool:
    """Whether `query` names the same thing as `name`: after normalising, one
    name's words are all among the other's ("NVIDIA" / "NVIDIA Corporation"),
    or the two are nearly identical (a typo). Merely similar-looking names
    ("Acme" / "A-MEM") do not match."""
    q, n = norm(query), norm(name)
    if not q or not n:
        return False
    q_words, n_words = set(q.split()), set(n.split())
    if q_words <= n_words or n_words <= q_words:
        return True
    return fuzz.ratio(q, n) >= threshold


def search_facts(
    store: Store,
    run_id: str,
    *,
    entity: str | None = None,
    relation: Relation | str | None = None,
    doc_id: str | None = None,
    entity_threshold: float = 0.85,
) -> list[QueryResult]:
    """Structured search: any filter left as `None` is not applied. Every
    returned fact has been re-verified against the canonical text (I1)."""
    graph = build_graph(store, run_id)
    relation_value = relation.value if isinstance(relation, Relation) else relation

    results: list[QueryResult] = []
    for u, v, _key, data in graph.edges(keys=True, data=True):
        if relation_value is not None and data["relation"] != relation_value:
            continue
        if doc_id is not None and data["doc_id"] != doc_id:
            continue
        if entity and not (
            names_match(entity, u, threshold=entity_threshold * 100)
            or names_match(entity, v, threshold=entity_threshold * 100)
        ):
            continue

        canonical = store.load_canonical_text(data["doc_id"])
        actual = canonical[data["evidence_start"] : data["evidence_end"]]
        if actual != data["evidence_text"]:
            # I1 re-verification failed: drop and log, never return with a warning (§6.6).
            print(
                f"[query] dropping stale fact: doc={data['doc_id']} "
                f"span=[{data['evidence_start']}:{data['evidence_end']}] no longer matches canonical text"
            )
            continue

        results.append(
            QueryResult(
                subject=u,
                relation=data["relation"],
                object=v,
                doc_id=data["doc_id"],
                evidence_text=data["evidence_text"],
                char_start=data["evidence_start"],
                char_end=data["evidence_end"],
                triple_id=_key,
            )
        )
    return results


def query_graph(
    store: Store, run_id: str, text: str, *, entity_threshold: float = 0.7
) -> list[QueryResult]:
    relation, entity_guess = parse_query(text)
    return search_facts(
        store, run_id, entity=entity_guess, relation=relation, entity_threshold=entity_threshold
    )
