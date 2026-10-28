"""Orchestrates the resolution cascade (§6.4): blocking -> filtering ->
adjudication (deferred band only) -> clustering -> entity write-back."""

from __future__ import annotations

from dataclasses import dataclass

from c2k.config import Config
from c2k.resolve import blocking, cluster, filtering
from c2k.resolve.adjudicate import adjudicate_pairs
from c2k.resolve.blocking import MentionRecord
from c2k.store import Store


@dataclass
class ResolveStats:
    n_mentions: int = 0
    n_candidate_pairs: int = 0
    n_accepted_direct: int = 0
    n_rejected: int = 0
    n_deferred: int = 0
    n_adjudicated_same: int = 0
    n_entities: int = 0

    @property
    def adjudication_rate(self) -> float | None:
        if self.n_candidate_pairs == 0:
            return None
        return self.n_deferred / self.n_candidate_pairs


def compute_resolution(
    mentions: list[MentionRecord],
    canonical_by_doc: dict[str, str],
    config: Config,
    run_id: str,
    *,
    embeddings=None,
) -> tuple[dict[int, int], dict[int, str], ResolveStats]:
    """The pure cascade (§6.4), with no DB writes: blocking -> filtering ->
    adjudication -> clustering. Returns (mention_index -> cluster_root,
    cluster_root -> canonical_name, stats). Split out from `resolve_run` so
    `c2k tune` can try several `(tau_lo, tau_hi)` values against the same
    mention set without touching the store — only the filtering step
    actually depends on the thresholds, so `embeddings` (and therefore
    blocking) can be computed once and reused across a sweep."""
    stats = ResolveStats()
    stats.n_mentions = len(mentions)
    if len(mentions) < 2:
        return {}, {}, stats

    if embeddings is None:
        embeddings = blocking.embed_mentions(mentions, canonical_by_doc, model_name=config.resolution.embedding_model)
    candidate_pairs = blocking.block_candidates(embeddings, top_k=config.resolution.top_k)
    stats.n_candidate_pairs = len(candidate_pairs)

    accepted_pairs: list[tuple[int, int]] = []
    deferred_pairs: list[tuple[int, int]] = []
    for i, j in candidate_pairs:
        decision, _score = filtering.filter_pair(
            mentions[i].surface,
            mentions[j].surface,
            tau_lo=config.resolution.tau_lo,
            tau_hi=config.resolution.tau_hi,
        )
        if decision is filtering.Decision.ACCEPT:
            accepted_pairs.append((i, j))
        elif decision is filtering.Decision.DEFER:
            deferred_pairs.append((i, j))
        # REJECT: dropped.

    stats.n_accepted_direct = len(accepted_pairs)
    stats.n_deferred = len(deferred_pairs)
    stats.n_rejected = stats.n_candidate_pairs - stats.n_accepted_direct - stats.n_deferred

    if deferred_pairs:
        text_pairs = [(mentions[i].surface, mentions[j].surface) for i, j in deferred_pairs]
        verdicts = adjudicate_pairs(text_pairs, run_id=run_id, batch_size=config.resolution.adjudication_batch)
        for (i, j), same in zip(deferred_pairs, verdicts):
            if same:
                accepted_pairs.append((i, j))
                stats.n_adjudicated_same += 1

    clusters = cluster.cluster_mentions(len(mentions), accepted_pairs)
    names = cluster.canonical_names([m.surface for m in mentions], clusters)
    stats.n_entities = len(names)
    return clusters, names, stats


def resolve_run(store: Store, config: Config, run_id: str) -> ResolveStats:
    rows = store.list_mentions(run_id)
    mentions = [
        MentionRecord(r["mention_id"], r["doc_id"], r["surface"], r["char_start"], r["char_end"])
        for r in rows
    ]
    if len(mentions) < 2:
        return ResolveStats(n_mentions=len(mentions))

    canonical_by_doc = {doc_id: store.load_canonical_text(doc_id) for doc_id in {m.doc_id for m in mentions}}
    clusters, names, stats = compute_resolution(mentions, canonical_by_doc, config, run_id)

    root_to_entity_id = {root: f"{run_id}:entity:{root}" for root in names}
    for idx, mention in enumerate(mentions):
        entity_id = root_to_entity_id[clusters[idx]]
        store.set_mention_entity(mention.mention_id, entity_id)
    for root, entity_id in root_to_entity_id.items():
        n_mentions_in_cluster = sum(1 for r in clusters.values() if r == root)
        store.insert_entity(entity_id, run_id, names[root], n_mentions_in_cluster)

    store.sync_triple_entities_from_mentions(run_id)
    return stats
