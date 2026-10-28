"""§6.6: NetworkX traversal + I1 re-verification. Query parsing is a plain
heuristic (no LLM), so these tests need no fake provider."""

from __future__ import annotations

from c2k.query import build_graph, parse_query, query_graph, search_facts
from c2k.schema import Relation, Triple
from c2k.store import Store


def _seed(store: Store) -> str:
    text = "Siemens acquired Mentor Graphics in 2017. Acme Corp is headquartered in Springfield."
    doc_id = "doc1"
    store.insert_document(doc_id, "doc1.pdf", "sha", len(text), "train_unused")
    store.save_canonical_text(doc_id, text)

    ev1 = "Siemens acquired Mentor Graphics"
    s1 = text.index(ev1)
    store.insert_mention("s1", "run1", doc_id, "Siemens", text.index("Siemens"), text.index("Siemens") + 7)
    store.insert_mention(
        "o1", "run1", doc_id, "Mentor Graphics", text.index("Mentor Graphics"), text.index("Mentor Graphics") + 15
    )
    store.insert_triple(
        Triple(
            triple_id="t1",
            run_id="run1",
            doc_id=doc_id,
            chunk_id="doc1:0",
            subject_mention_id="s1",
            object_mention_id="o1",
            relation=Relation.acquired,
            evidence_start=s1,
            evidence_end=s1 + len(ev1),
            evidence_text=ev1,
            confidence=0.9,
        )
    )

    ev2 = "Acme Corp is headquartered in Springfield"
    s2 = text.index(ev2)
    store.insert_mention("s2", "run1", doc_id, "Acme Corp", text.index("Acme Corp"), text.index("Acme Corp") + 9)
    store.insert_mention(
        "o2", "run1", doc_id, "Springfield", text.index("Springfield"), text.index("Springfield") + 11
    )
    store.insert_triple(
        Triple(
            triple_id="t2",
            run_id="run1",
            doc_id=doc_id,
            chunk_id="doc1:0",
            subject_mention_id="s2",
            object_mention_id="o2",
            relation=Relation.headquartered_in,
            evidence_start=s2,
            evidence_end=s2 + len(ev2),
            evidence_text=ev2,
            confidence=0.9,
        )
    )
    return text


def test_parse_query_finds_relation_and_entity() -> None:
    relation, entity = parse_query("which company is headquartered in Springfield")
    assert relation is Relation.headquartered_in
    assert entity == "Springfield"


def test_parse_query_underscore_relation_as_spaced_words() -> None:
    relation, _entity = parse_query("who acquired Mentor Graphics")
    assert relation is Relation.acquired


def test_parse_query_no_relation_keyword() -> None:
    relation, _entity = parse_query("tell me more")
    assert relation is None


def test_build_graph_has_both_edges(tmp_store: Store) -> None:
    _seed(tmp_store)
    graph = build_graph(tmp_store, "run1")
    assert graph.number_of_edges() == 2
    assert graph.has_edge("Siemens", "Mentor Graphics")
    assert graph.has_edge("Acme Corp", "Springfield")


def test_query_graph_filters_by_relation_and_entity(tmp_store: Store) -> None:
    _seed(tmp_store)
    results = query_graph(tmp_store, "run1", "who acquired Mentor Graphics")
    assert len(results) == 1
    assert results[0].subject == "Siemens"
    assert results[0].object == "Mentor Graphics"
    assert results[0].relation == "acquired"
    assert results[0].evidence_text == "Siemens acquired Mentor Graphics"


def test_query_graph_second_relation(tmp_store: Store) -> None:
    _seed(tmp_store)
    results = query_graph(tmp_store, "run1", "which company is headquartered in Springfield")
    assert len(results) == 1
    assert results[0].subject == "Acme Corp"
    assert results[0].object == "Springfield"


def test_query_graph_drops_facts_that_fail_i1_reverification(tmp_store: Store) -> None:
    _seed(tmp_store)
    # Simulate the canonical text changing under the stored span (should never
    # happen in practice, but §6.6 requires the check anyway).
    tmp_store.save_canonical_text("doc1", "completely different content now" * 5)
    results = query_graph(tmp_store, "run1", "who acquired Mentor Graphics")
    assert results == []


def test_query_graph_no_match_returns_empty(tmp_store: Store) -> None:
    _seed(tmp_store)
    results = query_graph(tmp_store, "run1", "who founded Nonexistent Corp")
    assert results == []


def test_search_facts_filters_by_relation_only(tmp_store: Store) -> None:
    _seed(tmp_store)
    results = search_facts(tmp_store, "run1", relation=Relation.acquired)
    assert [r.subject for r in results] == ["Siemens"]
    assert [r.subject for r in search_facts(tmp_store, "run1", relation="headquartered_in")] == ["Acme Corp"]


def test_search_facts_filters_by_entity_and_doc(tmp_store: Store) -> None:
    _seed(tmp_store)
    assert len(search_facts(tmp_store, "run1", entity="Springfield")) == 1
    assert len(search_facts(tmp_store, "run1", doc_id="doc1")) == 2
    assert search_facts(tmp_store, "run1", doc_id="no_such_doc") == []


def test_search_facts_no_filters_returns_everything_verified(tmp_store: Store) -> None:
    _seed(tmp_store)
    assert len(search_facts(tmp_store, "run1")) == 2
    tmp_store.save_canonical_text("doc1", "completely different content now" * 5)
    assert search_facts(tmp_store, "run1") == []


import pytest  # noqa: E402

from c2k.query import RELATION_PHRASES  # noqa: E402


@pytest.mark.parametrize(
    "question, relation, entity",
    [
        ("where is apple located?", Relation.headquartered_in, "apple"),
        ("Where is Apple based", Relation.headquartered_in, "Apple"),
        ("where is Walmart", Relation.headquartered_in, "Walmart"),
        ("who bought Hugging Face", Relation.acquired, "Hugging Face"),
        ("what did NVIDIA purchase", Relation.acquired, "NVIDIA"),
        ("who is the CEO of Disney", Relation.ceo_of, "Disney"),
        ("who runs Disney", Relation.ceo_of, "Disney"),
        ("who founded Armadin", Relation.founded_by, "Armadin"),
        ("which companies partner with NVIDIA", Relation.partnered_with, "NVIDIA"),
        ("who funds this research", Relation.funded_by, None),
        ("what was EGF tested on", Relation.evaluated_on, "EGF"),
        ("which models were evaluated on WearableQA", Relation.evaluated_on, "WearableQA"),
        ("which methods outperform DiGress", Relation.outperforms, "DiGress"),
        ("what does GPT-4o beat", Relation.outperforms, "GPT-4o"),
    ],
)
def test_parse_query_understands_paraphrases(question, relation, entity) -> None:
    assert parse_query(question) == (relation, entity)


def test_entity_alone_and_filler_alone() -> None:
    assert parse_query("apple") == (None, "apple")
    assert parse_query("tell me more") == (None, None)


def test_every_relation_has_everyday_phrasings() -> None:
    assert {r.value for r in Relation} == set(RELATION_PHRASES)
    assert all(RELATION_PHRASES[r.value] for r in Relation)


def test_phrase_file_with_an_unknown_relation_is_rejected(tmp_path, monkeypatch) -> None:
    import c2k.query as q

    bad = tmp_path / "phrases.yaml"
    bad.write_text("phrases:\n  not_a_relation: ['x']\nwhere_is: not_a_relation\n")
    monkeypatch.setattr(q, "RELATION_PHRASES_PATH", bad)
    with pytest.raises(ValueError, match="unknown relation"):
        q._load_relation_phrases()


from c2k.query import names_match  # noqa: E402


@pytest.mark.parametrize(
    "query, name, expected",
    [
        ("NVIDIA", "NVIDIA Corporation", True),
        ("apple", "Apple Inc.", True),
        ("Hugging Face", "The Hugging Face transformers library", True),
        ("walmrt", "Walmart Inc.", True),  # a typo
        ("up hugging face", "Hugging Face, Inc.", True),  # extra words in the question do not matter
        ("Acme Corp", "A-MEM", False),  # merely similar-looking names do not match
        ("Apple", "Pineapple", False),
        ("Disney", "Netflix", False),
    ],
)
def test_names_match_needs_real_overlap_or_a_near_identical_spelling(query, name, expected) -> None:
    assert names_match(query, name) is expected
