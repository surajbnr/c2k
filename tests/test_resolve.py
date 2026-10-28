"""§6.4: the resolution cascade — blocking -> filtering -> adjudication ->
clustering. Blocking's embedding step is monkeypatched to a deterministic
stand-in (loading `sentence-transformers` needs a network fetch of model
weights, which this suite can't assume); everything downstream — FAISS
top-k, rapidfuzz accept/reject/defer, the LLM adjudication call, and
union-find clustering — is real.
"""

from __future__ import annotations

import numpy as np
import pytest

from c2k import llm
from c2k import resolve as resolve_pipeline
from c2k.config import (
    ChunkingConfig,
    Config,
    EvaluationConfig,
    ExtractionConfig,
    ModelConfig,
    PathsConfig,
    ResolutionConfig,
)
from c2k.resolve import blocking, cluster
from c2k.resolve.adjudicate import BatchVerdicts, PairVerdict, adjudicate_batch, adjudicate_pairs
from c2k.resolve.filtering import Decision, decide, filter_pair, score_pair
from c2k.schema import Relation, Triple
from c2k.store import Store

# ---------------------------------------------------------------- filtering


def test_decide_boundaries() -> None:
    assert decide(0.71, tau_lo=0.72, tau_hi=0.88) is Decision.REJECT
    assert decide(0.72, tau_lo=0.72, tau_hi=0.88) is Decision.DEFER
    assert decide(0.88, tau_lo=0.72, tau_hi=0.88) is Decision.DEFER
    assert decide(0.89, tau_lo=0.72, tau_hi=0.88) is Decision.ACCEPT


def test_filter_pair_accepts_suffix_variant() -> None:
    decision, score = filter_pair("Siemens", "Siemens AG")
    assert decision is Decision.ACCEPT
    assert score == 1.0


def test_filter_pair_rejects_unrelated() -> None:
    decision, _score = filter_pair("Siemens", "Totally Unrelated Co")
    assert decision is Decision.REJECT


def test_filter_pair_defers_near_miss() -> None:
    decision, score = filter_pair("Mentor Graphics", "Mentor Grafiks")
    assert decision is Decision.DEFER
    assert 0.72 <= score <= 0.88


# ---------------------------------------------------------------- cluster


def test_cluster_mentions_union_find() -> None:
    clusters = cluster.cluster_mentions(5, [(0, 1), (2, 3)])
    assert clusters[0] == clusters[1]
    assert clusters[2] == clusters[3]
    assert clusters[4] not in (clusters[0], clusters[2])


def test_canonical_names_tie_broken_by_longest() -> None:
    surfaces = ["Siemens", "Siemens AG", "Mentor Graphics", "Mentor Grafiks", "Totally Unrelated Co"]
    clusters = cluster.cluster_mentions(5, [(0, 1), (2, 3)])
    names = cluster.canonical_names(surfaces, clusters)
    assert names[clusters[0]] == "Siemens AG"
    assert names[clusters[2]] == "Mentor Graphics"  # tie: longer of the two
    assert names[clusters[4]] == "Totally Unrelated Co"


# ---------------------------------------------------------------- blocking


def test_block_candidates_covers_all_pairs_when_k_large() -> None:
    embeddings = np.random.RandomState(0).rand(5, 8).astype("float32")
    pairs = blocking.block_candidates(embeddings, top_k=20)
    expected = {(i, j) for i in range(5) for j in range(i + 1, 5)}
    assert set(pairs) == expected


def test_block_candidates_no_self_pairs_and_deduped() -> None:
    embeddings = np.random.RandomState(1).rand(6, 4).astype("float32")
    pairs = blocking.block_candidates(embeddings, top_k=3)
    for i, j in pairs:
        assert i < j
    assert len(pairs) == len(set(pairs))


def test_mention_embedding_text_windows_context() -> None:
    canonical = "x" * 200 + "TARGET" + "y" * 200
    mention = blocking.MentionRecord("m1", "doc1", "TARGET", 200, 206)
    text = blocking.mention_embedding_text(mention, canonical)
    assert text.startswith("TARGET | ")
    context = text.split(" | ", 1)[1]
    assert len(context) == 100 + 6 + 100


# ---------------------------------------------------------------- adjudicate


@pytest.fixture
def fake_llm(monkeypatch):
    queue: list = []

    def _provider(prompt, response_model, *, model, temperature, max_retries):
        assert queue, "test queued no fake response"
        return queue.pop(0), {"input_tokens": 1, "output_tokens": 1}

    monkeypatch.setitem(llm._PROVIDERS, "fake", _provider)
    return queue


def _make_config() -> Config:
    return Config(
        arm="a2",
        model=ModelConfig(provider="fake", name="fake-model", temperature=0.0),
        chunking=ChunkingConfig(),
        extraction=ExtractionConfig(),
        resolution=ResolutionConfig(),
        evaluation=EvaluationConfig(),
        paths=PathsConfig(),
    )


def test_adjudicate_pairs_batches_and_maps_verdicts(tmp_store: Store, fake_llm) -> None:
    config = _make_config()
    llm.configure(tmp_store, config)
    tmp_store.start_run("run1", "a2", 0, "fake-model", "{}")

    fake_llm.append(BatchVerdicts(verdicts=[PairVerdict(same=True), PairVerdict(same=False)]))
    result = adjudicate_pairs([("A", "A Inc"), ("A", "Z")], run_id="run1", batch_size=20)
    assert result == [True, False]


def test_adjudicate_batch_size_mismatch_raises(tmp_store: Store, fake_llm) -> None:
    """The single-batch call still raises on a malformed response — same
    contract as llm.call() itself. Resilience against this lives one level
    up, in adjudicate_pairs (see below), not here."""
    config = _make_config()
    llm.configure(tmp_store, config)
    tmp_store.start_run("run1", "a2", 0, "fake-model", "{}")

    fake_llm.append(BatchVerdicts(verdicts=[PairVerdict(same=True)]))  # only 1, but 2 pairs sent
    with pytest.raises(RuntimeError):
        adjudicate_batch([("A", "B"), ("C", "D")], run_id="run1")


def test_adjudicate_pairs_survives_one_bad_batch(tmp_store: Store, fake_llm) -> None:
    """A batch that comes back malformed must not discard every other
    batch's already-adjudicated verdicts (found live: a small local model
    returned 1 verdict for a 20-pair batch mid-run, which used to crash the
    entire resolution step). The failed batch's pairs conservatively count
    as "not the same entity" rather than risking a wrong merge."""
    config = _make_config()
    llm.configure(tmp_store, config)
    tmp_store.start_run("run1", "a2", 0, "fake-model", "{}")

    # batch 1 (2 pairs) succeeds; batch 2 (2 pairs) comes back malformed;
    # batch 3 (2 pairs) succeeds again.
    fake_llm.append(BatchVerdicts(verdicts=[PairVerdict(same=True), PairVerdict(same=False)]))
    fake_llm.append(BatchVerdicts(verdicts=[PairVerdict(same=True)]))  # malformed: 1 for 2 pairs
    fake_llm.append(BatchVerdicts(verdicts=[PairVerdict(same=False), PairVerdict(same=True)]))

    pairs = [("A", "A2"), ("B", "B2"), ("C", "C2"), ("D", "D2"), ("E", "E2"), ("F", "F2")]
    result = adjudicate_pairs(pairs, run_id="run1", batch_size=2)

    assert result == [True, False, False, False, False, True]


# ---------------------------------------------------------------- resolve_run orchestration


def test_resolve_run_end_to_end(tmp_store: Store, fake_llm, monkeypatch) -> None:
    config = _make_config()
    llm.configure(tmp_store, config)

    doc_id = "doc1"
    tmp_store.insert_document(doc_id, "doc1.pdf", "sha", 500, "train_unused")
    tmp_store.save_canonical_text(doc_id, "x" * 500)

    surfaces = ["Siemens", "Siemens AG", "Mentor Graphics", "Mentor Grafiks", "Totally Unrelated Co"]
    mention_ids = [f"m{i}" for i in range(len(surfaces))]
    for i, (mid, surface) in enumerate(zip(mention_ids, surfaces)):
        tmp_store.insert_mention(mid, "run1", doc_id, surface, i * 10, i * 10 + len(surface))

    # Link a triple to two of the mentions so we can check entity sync too.
    tmp_store.insert_triple(
        Triple(
            triple_id="t1",
            run_id="run1",
            doc_id=doc_id,
            chunk_id="doc1:0",
            subject_mention_id="m0",
            object_mention_id="m2",
            relation=Relation.acquired,
            evidence_start=0,
            evidence_end=3,
            evidence_text="x" * 3,
            confidence=0.9,
        )
    )

    def fake_embed_mentions(mentions, canonical_by_doc, *, model_name):
        return np.eye(len(mentions), dtype="float32")

    monkeypatch.setattr(blocking, "embed_mentions", fake_embed_mentions)

    fake_llm.append(BatchVerdicts(verdicts=[PairVerdict(same=True, reason="typo variant")]))

    tmp_store.start_run("run1", "a2", 0, "fake-model", "{}")
    stats = resolve_pipeline.resolve_run(tmp_store, config, "run1")

    assert stats.n_mentions == 5
    assert stats.n_candidate_pairs == 10  # C(5,2), top_k=20 covers everyone
    assert stats.n_accepted_direct == 1  # (Siemens, Siemens AG)
    assert stats.n_deferred == 1  # (Mentor Graphics, Mentor Grafiks)
    assert stats.n_adjudicated_same == 1
    assert stats.n_entities == 3
    assert stats.adjudication_rate == pytest.approx(0.1)

    entity_ids = {row["mention_id"]: row["entity_id"] for row in tmp_store.list_mentions("run1")}
    assert entity_ids["m0"] == entity_ids["m1"]
    assert entity_ids["m2"] == entity_ids["m3"]
    assert entity_ids["m4"] not in (entity_ids["m0"], entity_ids["m2"])

    triple_row = tmp_store.list_triples("run1")[0]
    assert triple_row["subject_entity_id"] == entity_ids["m0"]
    assert triple_row["object_entity_id"] == entity_ids["m2"]


def test_resolve_run_skips_when_fewer_than_two_mentions(tmp_store: Store) -> None:
    config = _make_config()
    tmp_store.start_run("run1", "a2", 0, "fake-model", "{}")
    stats = resolve_pipeline.resolve_run(tmp_store, config, "run1")
    assert stats.n_mentions == 0
    assert stats.n_candidate_pairs == 0
