"""All metric computations (§9.2). Every function here takes plain data
(counts, EvalTriple lists, verdict lists) — no module in this file imports
`c2k.store` at call time for the pure-math functions, which is what makes
`test_metrics.py`'s hand-checked fixture possible without a database.

`confidence` is model-reported and uncalibrated (§6.3): no function here
reads it.
"""

from __future__ import annotations

from dataclasses import dataclass

from c2k.eval.entailment import cohens_kappa
from c2k.eval.match import EvalTriple, match, sim
from c2k.norm import norm
from c2k.schema import Relation
from c2k.store import Split, Store, TestUnlock


@dataclass(frozen=True)
class PRF1:
    precision: float
    recall: float
    f1: float


def precision_recall_f1(n_matched: int, n_predicted: int, n_gold: int) -> PRF1:
    precision = n_matched / n_predicted if n_predicted else 0.0
    recall = n_matched / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return PRF1(precision, recall, f1)


def triple_prf1(
    predicted: list[EvalTriple],
    gold: list[EvalTriple],
    *,
    entity_threshold: float = 0.85,
    span_iou_threshold: float | None = None,
    n_unmatchable_predicted: int = 0,
) -> PRF1:
    """§8: A0's free-form predicates that failed post-hoc mapping to the
    closed vocabulary never became a `Triple` (I3 forbids a non-enum
    relation), so they can't appear in `predicted` — but they were still a
    prediction the model made, and §8 says "unmapped predicates count as
    misses". `n_unmatchable_predicted` folds them into the precision
    denominator only: they can never *match* gold (recall is about gold
    coverage, unaffected), but they inflate precision if silently dropped.
    """
    matches = match(
        predicted, gold, entity_threshold=entity_threshold, span_iou_threshold=span_iou_threshold
    )
    n_predicted = len(predicted) + n_unmatchable_predicted
    return precision_recall_f1(len(matches), n_predicted, len(gold))


def hallucination_rate(verdicts: list[bool]) -> float | None:
    """Fraction of a judged sample whose evidence span does NOT entail the
    triple. `verdicts[i] == True` means "entails". Returns None when nothing
    has been judged yet (no sample recorded)."""
    if not verdicts:
        return None
    n_bad = sum(1 for v in verdicts if not v)
    return n_bad / len(verdicts)


def er_pairwise_f1(
    predicted_clusters: dict[str, str], gold_clusters: dict[str, str]
) -> PRF1 | None:
    """§9.2: over all mention pairs in gold docs, predicted-same vs
    gold-same. Both args map mention_id -> cluster_id. Returns None if either
    side has no clustering (i.e. resolution has not run)."""
    if not predicted_clusters or not gold_clusters:
        return None
    mentions = sorted(set(predicted_clusters) & set(gold_clusters))
    n_matched = n_pred_pos = n_gold_pos = 0
    for i in range(len(mentions)):
        for j in range(i + 1, len(mentions)):
            a, b = mentions[i], mentions[j]
            pred_same = predicted_clusters[a] == predicted_clusters[b]
            gold_same = gold_clusters[a] == gold_clusters[b]
            if pred_same:
                n_pred_pos += 1
            if gold_same:
                n_gold_pos += 1
            if pred_same and gold_same:
                n_matched += 1
    return precision_recall_f1(n_matched, n_pred_pos, n_gold_pos)


def predicted_clusters_from_store(
    store: Store, run_id: str, doc_ids: set[str] | None = None
) -> dict[str, str]:
    """mention_id -> entity_id, for `er_pairwise_f1`'s predicted side.
    Restrict to `doc_ids` (§9.2: ER F1 is scored "over all mention pairs in
    gold docs"). Mentions resolution never assigned an entity to — resolution
    didn't run, or the run had fewer than 2 mentions total — are omitted."""
    rows = store.list_mentions(run_id)
    return {
        r["mention_id"]: r["entity_id"]
        for r in rows
        if r["entity_id"] is not None and (doc_ids is None or r["doc_id"] in doc_ids)
    }


def gold_clusters_for_mentions(
    mention_rows: list, gold_rows: list, *, entity_threshold: float = 0.85
) -> dict[str, str]:
    """Derives a "gold-same" clustering for ER pairwise F1 (§9.2) from gold
    triples' raw subject/object strings. The gold JSONL format (§10) carries
    no coreference annotation of its own — this is the only derivation it
    admits: a mention is assigned to whichever gold entity string in the same
    document it's closest to, at `entity_threshold`. A mention matching no
    gold string within threshold is omitted, so it contributes no pairs
    (`er_pairwise_f1` intersects the predicted/gold key sets already)."""
    gold_strings_by_doc: dict[str, list[str]] = {}
    for g in gold_rows:
        gold_strings_by_doc.setdefault(g["doc_id"], []).extend([g["subject"], g["object"]])

    clusters: dict[str, str] = {}
    for m in mention_rows:
        best_score, best_string = 0.0, None
        for g_string in gold_strings_by_doc.get(m["doc_id"], []):
            score = sim(norm(m["surface"]), norm(g_string))
            if score > best_score:
                best_score, best_string = score, g_string
        if best_string is not None and best_score >= entity_threshold:
            clusters[m["mention_id"]] = f"{m['doc_id']}:{norm(best_string)}"
    return clusters


def schema_recall_ceiling(n_out_of_schema: int, n_total_gold: int) -> float | None:
    if n_total_gold == 0:
        return None
    return 1 - (n_out_of_schema / n_total_gold)


def span_localization_failure_rate(n_dropped: int, n_emitted: int) -> float | None:
    total = n_dropped + n_emitted
    if total == 0:
        return None
    return n_dropped / total


def adjudication_rate(n_adjudicated: int, n_candidate_pairs: int) -> float | None:
    if n_candidate_pairs == 0:
        return None
    return n_adjudicated / n_candidate_pairs


@dataclass(frozen=True)
class CostSummary:
    n_calls: int
    n_input_tokens: int
    n_output_tokens: int
    wall_seconds: float
    n_documents: int

    @property
    def calls_per_doc(self) -> float:
        return self.n_calls / self.n_documents if self.n_documents else 0.0

    @property
    def input_tokens_per_doc(self) -> float:
        return self.n_input_tokens / self.n_documents if self.n_documents else 0.0

    @property
    def output_tokens_per_doc(self) -> float:
        return self.n_output_tokens / self.n_documents if self.n_documents else 0.0

    @property
    def wall_seconds_per_doc(self) -> float:
        return self.wall_seconds / self.n_documents if self.n_documents else 0.0


# ---------------------------------------------------------------- DB loaders


def load_predicted(store: Store, run_id: str, *, split: Split | None = None) -> list[EvalTriple]:
    """With `split`, keeps only triples from documents assigned to that split:
    gold only exists for the split being scored, so predictions from other
    splits' documents could never match and would just deflate precision."""
    rows = store.list_triples_with_surface(run_id)
    if split is not None:
        split_docs = {d["doc_id"] for d in store.list_documents() if d["split"] == split.value}
        rows = [r for r in rows if r["doc_id"] in split_docs]
    return [
        EvalTriple(
            id=r["triple_id"],
            subject=r["subject_surface"],
            relation=Relation(r["relation"]),
            object=r["object_surface"],
            evidence_start=r["evidence_start"],
            evidence_end=r["evidence_end"],
        )
        for r in rows
    ]


def load_gold(
    store: Store, split: Split, *, unlock: TestUnlock | None = None
) -> tuple[list[EvalTriple], int]:
    """Returns (in_schema_gold, n_total_gold_including_out_of_schema)."""
    rows = store.load_gold(split, unlock=unlock)
    in_schema = [
        EvalTriple(
            id=r["gold_id"],
            subject=r["subject"],
            relation=Relation(r["relation"]),
            object=r["object"],
            evidence_start=r["evidence_start"],
            evidence_end=r["evidence_end"],
        )
        for r in rows
        if not r["out_of_schema"]
    ]
    return in_schema, len(rows)


def cost_summary(store: Store, run_id: str) -> CostSummary:
    run = store.get_run(run_id)
    if run is None:
        raise KeyError(f"no such run: {run_id}")
    doc_ids = {r["doc_id"] for r in store.list_triples(run_id)}
    return CostSummary(
        n_calls=run["n_calls"],
        n_input_tokens=run["n_input_tokens"],
        n_output_tokens=run["n_output_tokens"],
        wall_seconds=run["wall_seconds"],
        n_documents=len(doc_ids) or 1,
    )


@dataclass
class RunMetrics:
    run_id: str
    triple_loose: PRF1
    triple_span_strict: PRF1
    hallucination: float | None
    hallucination_llm: float | None
    entailment_kappa: float | None
    er_pairwise: PRF1 | None
    schema_recall_ceiling: float | None
    span_localization_failure_rate: float | None
    adjudication_rate: float | None
    cost: CostSummary
    n_predicted: int
    n_gold_in_schema: int
    n_gold_total: int


def evaluate_run(
    store: Store,
    run_id: str,
    split: Split,
    *,
    unlock: TestUnlock | None = None,
    entity_threshold: float = 0.85,
    span_iou_threshold: float = 0.5,
    entailment_verdicts: list[bool] | None = None,
    entailment_verdicts_llm: list[bool] | None = None,
    n_span_failures: int | None = None,
    n_adjudicated: int | None = None,
    n_candidate_pairs: int | None = None,
    n_unmatchable_predicted: int | None = None,
    predicted_clusters: dict[str, str] | None = None,
    gold_clusters: dict[str, str] | None = None,
) -> RunMetrics:
    """Any count/verdict/cluster kwarg left as `None` is loaded from the
    store instead of defaulting to 0/empty — §7's "do not silently return
    partial results" applies to the evaluator's own inputs too: a pipeline
    stage that never ran should surface as `n/a`, not as a fabricated zero.
    Passing an explicit value (including `[]` or `{}`) always wins, which is
    what lets `test_metrics.py` exercise this purely, with no store at all.
    """
    predicted = load_predicted(store, run_id, split=split)
    gold_in_schema, n_gold_total = load_gold(store, split, unlock=unlock)

    run_stats = store.get_run_stats(run_id)

    if n_span_failures is None:
        n_span_failures = run_stats["n_span_localization_failures"] if run_stats else None
    if n_unmatchable_predicted is None:
        n_unmatchable_predicted = (run_stats["n_unmapped_relations"] if run_stats else None) or 0
    if n_candidate_pairs is None:
        n_candidate_pairs = run_stats["n_candidate_pairs"] if run_stats else None
    if n_adjudicated is None:
        # §9.2: "LLM-adjudicated pairs" are every pair that landed in the
        # deferred band, since `compute_resolution` sends all of them to the
        # LLM (`c2k/resolve/__init__.py`) — that's `n_deferred`, not
        # `n_adjudicated_same` (which only counts the ones the LLM merged).
        n_adjudicated = run_stats["n_deferred"] if run_stats else None

    if entailment_verdicts is None:
        human_rows = store.get_entailment_judgements(run_id, "human")
        entailment_verdicts = [bool(r["verdict"]) for r in human_rows]
    if entailment_verdicts_llm is None:
        llm_rows = store.get_entailment_judgements(run_id, "llm")
        entailment_verdicts_llm = [bool(r["verdict"]) for r in llm_rows]

    if predicted_clusters is None or gold_clusters is None:
        gold_doc_ids = {r["doc_id"] for r in store.load_gold(split, unlock=unlock)}
        if predicted_clusters is None:
            predicted_clusters = predicted_clusters_from_store(store, run_id, gold_doc_ids)
        if gold_clusters is None:
            mention_rows = [r for r in store.list_mentions(run_id) if r["doc_id"] in gold_doc_ids]
            gold_rows = store.load_gold(split, unlock=unlock)
            gold_clusters = gold_clusters_for_mentions(
                mention_rows, gold_rows, entity_threshold=entity_threshold
            )

    kappa = None
    if entailment_verdicts and entailment_verdicts_llm and len(entailment_verdicts) == len(entailment_verdicts_llm):
        kappa = cohens_kappa(entailment_verdicts, entailment_verdicts_llm)

    return RunMetrics(
        run_id=run_id,
        triple_loose=triple_prf1(
            predicted,
            gold_in_schema,
            entity_threshold=entity_threshold,
            n_unmatchable_predicted=n_unmatchable_predicted,
        ),
        triple_span_strict=triple_prf1(
            predicted,
            gold_in_schema,
            entity_threshold=entity_threshold,
            span_iou_threshold=span_iou_threshold,
            n_unmatchable_predicted=n_unmatchable_predicted,
        ),
        hallucination=hallucination_rate(entailment_verdicts),
        hallucination_llm=hallucination_rate(entailment_verdicts_llm),
        entailment_kappa=kappa,
        er_pairwise=er_pairwise_f1(predicted_clusters, gold_clusters),
        schema_recall_ceiling=schema_recall_ceiling(
            n_gold_total - len(gold_in_schema), n_gold_total
        ),
        span_localization_failure_rate=span_localization_failure_rate(
            n_span_failures, len(predicted)
        )
        if n_span_failures is not None
        else None,
        adjudication_rate=adjudication_rate(n_adjudicated, n_candidate_pairs)
        if n_adjudicated is not None and n_candidate_pairs is not None
        else None,
        cost=cost_summary(store, run_id),
        n_predicted=len(predicted),
        n_gold_in_schema=len(gold_in_schema),
        n_gold_total=n_gold_total,
    )
