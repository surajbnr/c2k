"""§11's CLI, backed by Typer.

`TestUnlock` (I5) is constructed only in this module, and only inside `eval`
and `ablate` — the two commands where a human explicitly named the test
split on the command line (`--split test`, or `ablate` itself, which §13
mandates runs "on the test split" from D31 onward). `tune` never constructs
one (`tests/test_split_lock.py` checks this by name), so no tuning or
threshold-sweep code path can reach test-split gold.
"""

from __future__ import annotations

import getpass
import json
from pathlib import Path

import typer
from pydantic import BaseModel

from c2k import chunk as chunk_mod
from c2k import extract as extract_mod
from c2k import ingest as ingest_mod
from c2k import llm
from c2k import query as query_mod
from c2k import resolve as resolve_pipeline
from c2k.config import Config, apply_arm, load_config
from c2k.eval import entailment as eval_entailment
from c2k.eval import metrics as eval_metrics
from c2k.eval import report as eval_report
from c2k.schema import Relation, Triple
from c2k.store import Split, Store, TestUnlock

app = typer.Typer(help="C2K — schema-constrained knowledge extraction pipeline.", add_completion=False)


def _canonical_dir(config: Config) -> Path:
    """§4 doesn't name a location for the persisted canonical text stream
    (§6.1); we keep it as a sibling of `pdfs/`/`gold/`/`db/` under the same
    data root implied by `paths.db`."""
    return Path(config.paths.db).parent.parent / "canonical"


def _store_from_config(config: Config) -> Store:
    store = Store(config.paths.db, canonical_dir=_canonical_dir(config))
    store.initialize()
    return store


@app.command()
def ingest(input: Path = typer.Option(Path("data/pdfs"), "--input")) -> None:
    """PDFs -> `documents` rows + canonical text (§6.1)."""
    config = load_config()
    store = _store_from_config(config)
    doc_ids = ingest_mod.ingest_directory(store, input)
    n_gold = store.sync_gold_from_jsonl(config.paths.gold)
    typer.echo(
        f"ingested {len(doc_ids)} document(s): {', '.join(doc_ids) or '(none found)'}; "
        f"synced {n_gold} gold fact(s) from {config.paths.gold}"
    )


@app.command()
def chunk(doc_id: str = typer.Option("all", "--doc-id")) -> None:
    """Canonical text -> overlapping chunks (§6.2)."""
    config = load_config()
    store = _store_from_config(config)
    doc_ids = [r["doc_id"] for r in store.list_documents()] if doc_id == "all" else [doc_id]
    total = 0
    for d in doc_ids:
        chunks = chunk_mod.chunk_document(
            store,
            d,
            tokens=config.chunking.tokens,
            overlap=config.chunking.overlap,
            snap_window=config.chunking.snap_window,
        )
        total += len(chunks)
    typer.echo(f"chunked {len(doc_ids)} document(s) into {total} chunk(s)")


@app.command()
def extract(
    arm: str = typer.Option(..., "--arm"),
    repeat: int = typer.Option(0, "--repeat"),
) -> None:
    """Chunk -> List[Triple] (§6.3), one `llm.call()` per chunk."""
    config = apply_arm(load_config(), arm)
    store = _store_from_config(config)
    llm.configure(store, config)

    run_id = f"{arm}_r{repeat}"
    store.start_run(run_id, arm, repeat, config.model.name, config.model_dump_json())
    stats = extract_mod.extract_corpus(store, config, run_id)
    store.record_extraction_stats(
        run_id,
        n_chunks=stats.n_chunks,
        n_extracted=stats.n_extracted,
        n_stored=stats.n_stored,
        n_span_localization_failures=stats.n_span_localization_failures,
        n_deduplicated=stats.n_deduplicated,
        n_unmapped_relations=stats.n_unmapped_relations,
        n_llm_failures=stats.n_llm_failures,
    )
    store.finish_run(run_id)

    typer.echo(
        f"run {run_id!r}: {stats.n_chunks} chunk(s) -> {stats.n_extracted} extracted, "
        f"{stats.n_stored} stored, {stats.n_span_localization_failures} span-localization "
        f"failure(s), {stats.n_deduplicated} deduplicated, {stats.n_unmapped_relations} "
        f"unmapped relation(s), {stats.n_llm_failures} LLM call failure(s)"
    )


@app.command()
def resolve(run_id: str = typer.Option(..., "--run-id")) -> None:
    """Mention resolution cascade: blocking -> filtering -> adjudication ->
    clustering (§6.4)."""
    config = load_config()
    store = _store_from_config(config)
    llm.configure(store, config)

    stats = resolve_pipeline.resolve_run(store, config, run_id)
    store.record_resolution_stats(
        run_id,
        n_mentions=stats.n_mentions,
        n_candidate_pairs=stats.n_candidate_pairs,
        n_accepted_direct=stats.n_accepted_direct,
        n_rejected=stats.n_rejected,
        n_deferred=stats.n_deferred,
        n_adjudicated_same=stats.n_adjudicated_same,
        n_entities=stats.n_entities,
    )
    typer.echo(
        f"run {run_id!r}: {stats.n_mentions} mention(s), {stats.n_candidate_pairs} candidate "
        f"pair(s) -> {stats.n_accepted_direct} accepted, {stats.n_rejected} rejected, "
        f"{stats.n_deferred} deferred ({stats.n_adjudicated_same} adjudicated same) -> "
        f"{stats.n_entities} entit{'y' if stats.n_entities == 1 else 'ies'}"
    )


@app.command(name="eval")
def eval_cmd(
    run_id: str = typer.Option(..., "--run-id"),
    split: str = typer.Option(..., "--split"),
) -> None:
    """Runs the evaluator (§9) against a stored run. This is the only place
    in the codebase that may construct a `TestUnlock` — it does so only when
    the caller literally passed `--split test`."""
    config = load_config()
    store = _store_from_config(config)
    split_enum = Split(split)
    unlock = TestUnlock(reason="cli --split test") if split_enum is Split.TEST else None
    metrics = eval_metrics.evaluate_run(
        store,
        run_id,
        split_enum,
        unlock=unlock,
        entity_threshold=config.evaluation.entity_match_threshold,
        span_iou_threshold=config.evaluation.span_iou_threshold,
    )
    typer.echo(eval_report.render_run(metrics))


@app.command()
def judge(
    run_id: str = typer.Option(..., "--run-id"),
    n: int = typer.Option(100, "--n"),
) -> None:
    """Human entailment labelling (§9.3) — the headline hallucination number.
    Also runs the LLM judge over the same sample and reports it alongside
    with Cohen's kappa, per §9.3's "reported alongside... a finding in
    itself" — not the headline, but not skippable either."""
    config = load_config()
    store = _store_from_config(config)
    llm.configure(store, config)
    rows = store.list_triples_with_surface(run_id)
    triples = [
        {
            "triple_id": r["triple_id"],
            "subject": r["subject_surface"],
            "relation": r["relation"],
            "object": r["object_surface"],
            "evidence_text": r["evidence_text"],
        }
        for r in rows
    ]
    sample = eval_entailment.sample_triples_for_judging(triples, min(n, len(triples)))
    human_verdicts = eval_entailment.judge_human(store, run_id, sample)
    llm_verdicts = eval_entailment.judge_llm(store, run_id, sample)

    human_rate = eval_metrics.hallucination_rate(human_verdicts)
    llm_rate = eval_metrics.hallucination_rate(llm_verdicts)
    kappa = eval_entailment.cohens_kappa(human_verdicts, llm_verdicts)
    typer.echo(
        f"recorded {len(human_verdicts)} verdict(s). "
        f"hallucination rate: human={human_rate}, llm={llm_rate}, kappa={kappa}"
    )


@app.command()
def ablate(
    arms: str = typer.Option(..., "--arms"),
    repeats: int = typer.Option(3, "--repeats"),
) -> None:
    """Full sweep -> results table (§8). Runs on the test split (§13, D31) —
    this is the second and last place `TestUnlock` may be constructed (I5):
    ablation is, by definition, the final measurement, not tuning."""
    base_config = load_config()
    store = _store_from_config(base_config)
    arm_list = [a.strip() for a in arms.split(",") if a.strip()]

    per_arm: dict[str, list[eval_metrics.RunMetrics]] = {}
    for arm in arm_list:
        config = apply_arm(base_config, arm)
        llm.configure(store, config)
        per_repeat: list[eval_metrics.RunMetrics] = []
        for repeat in range(repeats):
            run_id = f"{arm}_r{repeat}"
            store.start_run(run_id, arm, repeat, config.model.name, config.model_dump_json())
            extract_stats = extract_mod.extract_corpus(store, config, run_id)
            store.record_extraction_stats(
                run_id,
                n_chunks=extract_stats.n_chunks,
                n_extracted=extract_stats.n_extracted,
                n_stored=extract_stats.n_stored,
                n_span_localization_failures=extract_stats.n_span_localization_failures,
                n_deduplicated=extract_stats.n_deduplicated,
                n_unmapped_relations=extract_stats.n_unmapped_relations,
                n_llm_failures=extract_stats.n_llm_failures,
            )
            if config.resolution.enabled:
                resolve_stats = resolve_pipeline.resolve_run(store, config, run_id)
                store.record_resolution_stats(
                    run_id,
                    n_mentions=resolve_stats.n_mentions,
                    n_candidate_pairs=resolve_stats.n_candidate_pairs,
                    n_accepted_direct=resolve_stats.n_accepted_direct,
                    n_rejected=resolve_stats.n_rejected,
                    n_deferred=resolve_stats.n_deferred,
                    n_adjudicated_same=resolve_stats.n_adjudicated_same,
                    n_entities=resolve_stats.n_entities,
                )
            store.finish_run(run_id)

            metrics = eval_metrics.evaluate_run(
                store,
                run_id,
                Split.TEST,
                unlock=TestUnlock(reason="cli ablate — final measurement, always test split"),
                entity_threshold=config.evaluation.entity_match_threshold,
                span_iou_threshold=config.evaluation.span_iou_threshold,
            )
            per_repeat.append(metrics)
        per_arm[arm] = per_repeat
        typer.echo(eval_report.render_arm_summary(arm, per_repeat))

    if "a0" in per_arm and "a2" in per_arm:
        typer.echo("")
        typer.echo(eval_report.render_success_criterion(per_arm["a0"], per_arm["a2"]))


_TAU_LO_GRID = (0.60, 0.66, 0.72, 0.78)
_TAU_HI_GRID = (0.82, 0.88, 0.94)


@app.command()
def tune(
    split: str = typer.Option("tune", "--split"),
    run_id: str = typer.Option(..., "--run-id"),
) -> None:
    """Threshold sweep over tau_lo/tau_hi (§6.4), tune split only (I5).
    Never constructs a `TestUnlock` and, unlike `eval`/`ablate`, structurally
    cannot: this function never imports the name (see
    `tests/test_split_lock.py::test_tune_command_source_never_names_test_unlock`).

    Uses an existing run's mentions restricted to tune-split documents,
    re-clusters them under each `(tau_lo, tau_hi)` in a small grid (embeddings
    computed once and reused — only filtering depends on the thresholds), and
    scores the resulting resolved triples against tune gold by loose triple
    F1. This never writes `config/default.yaml`; freezing a chosen threshold
    there is a human decision (the file comment already says so)."""
    split_enum = Split(split)
    if split_enum is not Split.TUNE:
        typer.echo("c2k tune only ever reads the tune split (I5) — refusing.", err=True)
        raise typer.Exit(code=1)

    config = load_config()
    store = _store_from_config(config)
    llm.configure(store, config)

    tune_doc_ids = {r["doc_id"] for r in store.list_documents() if r["split"] == Split.TUNE.value}
    if not tune_doc_ids:
        typer.echo("no documents assigned to the tune split — nothing to sweep.", err=True)
        raise typer.Exit(code=1)

    mention_rows = [r for r in store.list_mentions(run_id) if r["doc_id"] in tune_doc_ids]
    if len(mention_rows) < 2:
        typer.echo(f"fewer than 2 mentions in run {run_id!r} over tune-split docs — nothing to sweep.", err=True)
        raise typer.Exit(code=1)

    from c2k.resolve import compute_resolution
    from c2k.resolve.blocking import MentionRecord, embed_mentions

    mentions = [
        MentionRecord(r["mention_id"], r["doc_id"], r["surface"], r["char_start"], r["char_end"])
        for r in mention_rows
    ]
    mention_index = {m.mention_id: i for i, m in enumerate(mentions)}
    canonical_by_doc = {doc_id: store.load_canonical_text(doc_id) for doc_id in tune_doc_ids}
    embeddings = embed_mentions(mentions, canonical_by_doc, model_name=config.resolution.embedding_model)

    triple_rows = [
        r
        for r in store.list_triples(run_id)
        if r["doc_id"] in tune_doc_ids
        and r["subject_mention_id"] in mention_index
        and r["object_mention_id"] in mention_index
    ]
    gold, _total = eval_metrics.load_gold(store, Split.TUNE)

    sweep_run_id = f"{run_id}:tune-sweep"
    store.start_run(sweep_run_id, "tune", 0, config.model.name, config.model_dump_json())

    grid = [(lo, hi) for lo in _TAU_LO_GRID for hi in _TAU_HI_GRID if lo < hi]
    results = []
    for tau_lo, tau_hi in grid:
        trial_config = config.model_copy(
            update={"resolution": config.resolution.model_copy(update={"tau_lo": tau_lo, "tau_hi": tau_hi})}
        )
        clusters, names, stats = compute_resolution(
            mentions, canonical_by_doc, trial_config, sweep_run_id, embeddings=embeddings
        )
        predicted = [
            eval_metrics.EvalTriple(
                id=r["triple_id"],
                subject=names[clusters[mention_index[r["subject_mention_id"]]]],
                relation=Relation(r["relation"]),
                object=names[clusters[mention_index[r["object_mention_id"]]]],
                evidence_start=r["evidence_start"],
                evidence_end=r["evidence_end"],
            )
            for r in triple_rows
        ]
        prf1 = eval_metrics.triple_prf1(predicted, gold, entity_threshold=config.evaluation.entity_match_threshold)
        results.append((tau_lo, tau_hi, prf1, stats.adjudication_rate))
    store.finish_run(sweep_run_id)

    results.sort(key=lambda r: r[2].f1, reverse=True)
    typer.echo(
        f"threshold sweep: {len(mention_rows)} mention(s) / {len(triple_rows)} triple(s) from "
        f"run {run_id!r} over {len(tune_doc_ids)} tune-split doc(s)"
    )
    typer.echo(f"{'tau_lo':>8} {'tau_hi':>8} {'triple F1':>10} {'adjud. rate':>12}")
    for tau_lo, tau_hi, prf1, adjud in results:
        adjud_str = f"{adjud * 100:.1f}%" if adjud is not None else "n/a"
        typer.echo(f"{tau_lo:8.2f} {tau_hi:8.2f} {prf1.f1 * 100:9.1f}% {adjud_str:>12}")

    best_lo, best_hi, best_prf1, _ = results[0]
    typer.echo(
        f"\nbest: tau_lo={best_lo}, tau_hi={best_hi} (triple F1={best_prf1.f1 * 100:.1f}%). "
        "This does not modify config/default.yaml — freeze it there by hand if adopted."
    )


@app.command()
def query(
    text: str = typer.Argument(...),
    run_id: str = typer.Option(..., "--run-id"),
) -> None:
    """NetworkX traversal + span re-verification (§6.6). SPEC.md §11's
    example omits `--run-id`, but every other command that reads triples
    (`eval`, `judge`, `resolve`) requires one — a query has to name which
    run's graph it's traversing, same as those."""
    config = load_config()
    store = _store_from_config(config)
    results = query_mod.query_graph(store, run_id, text)
    if not results:
        typer.echo("no matching facts found.")
        return
    for r in results:
        typer.echo(f"{r.subject} --[{r.relation}]--> {r.object}")
        typer.echo(f"    {r.doc_id}[{r.char_start}:{r.char_end}]: {r.evidence_text!r}")


@app.command()
def annotate(doc_id: str = typer.Option(..., "--doc-id")) -> None:
    """Appends hand-annotated gold facts to `data/gold/{doc_id}.jsonl` (§10)."""
    config = load_config()
    store = _store_from_config(config)
    text = store.load_canonical_text(doc_id)
    gold_path = Path(config.paths.gold) / f"{doc_id}.jsonl"
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    annotator = getpass.getuser()

    typer.echo(f"canonical text for {doc_id}: {len(text)} chars. Blank subject to stop.")
    with gold_path.open("a", encoding="utf-8") as f:
        while True:
            subject = typer.prompt("subject", default="", show_default=False)
            if not subject:
                break
            out_of_schema = typer.confirm("out of schema (not in relations.yaml)?", default=False)
            relation = None if out_of_schema else typer.prompt("relation")
            obj = typer.prompt("object")
            start = typer.prompt("evidence_start", type=int)
            end = typer.prompt("evidence_end", type=int)
            typer.echo(f"  evidence[{start}:{end}] = {text[start:end]!r}")
            if not typer.confirm("append this entry?", default=True):
                continue
            entry = {
                "subject": subject,
                "relation": relation,
                "object": obj,
                "evidence_start": None if out_of_schema else start,
                "evidence_end": None if out_of_schema else end,
                "out_of_schema": out_of_schema,
                "annotator": annotator,
            }
            f.write(json.dumps(entry) + "\n")
            typer.echo("appended.")


class StubPrediction(BaseModel):
    """The row shape `c2k stub-run` reads — deliberately simpler than the
    stored `Triple` (raw subject/object strings, no mention/entity
    resolution) so the evaluator's D14 gate is exercisable with zero PDFs and
    no API key."""

    doc_id: str
    subject: str
    relation: Relation
    object: str
    evidence_start: int
    evidence_end: int
    evidence_text: str
    confidence: float = 1.0


@app.command(name="stub-run")
def stub_run(
    from_file: Path = typer.Option(..., "--from"),
    run_id: str = typer.Option("stub", "--run-id"),
    arm: str = typer.Option("stub", "--arm"),
) -> None:
    """Loads a JSONL of `StubPrediction` rows as a real run, so `c2k eval`
    can be demonstrated end-to-end before extract.py exists (D14, §13)."""
    config = load_config()
    store = _store_from_config(config)
    store.start_run(run_id, arm, 0, "stub", config.model_dump_json())

    n = 0
    with from_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pred = StubPrediction.model_validate_json(line)
            triple_id = f"{run_id}:{n}"
            subj_mention_id, obj_mention_id = f"{triple_id}:subj", f"{triple_id}:obj"
            # Stub simplification: both mentions reuse the triple's evidence
            # span, since stub predictions carry no separate subject/object offsets.
            store.insert_mention(
                subj_mention_id, run_id, pred.doc_id, pred.subject, pred.evidence_start, pred.evidence_end
            )
            store.insert_mention(
                obj_mention_id, run_id, pred.doc_id, pred.object, pred.evidence_start, pred.evidence_end
            )
            triple = Triple(
                triple_id=triple_id,
                run_id=run_id,
                doc_id=pred.doc_id,
                chunk_id="stub",
                subject_mention_id=subj_mention_id,
                object_mention_id=obj_mention_id,
                relation=pred.relation,
                evidence_start=pred.evidence_start,
                evidence_end=pred.evidence_end,
                evidence_text=pred.evidence_text,
                confidence=pred.confidence,
            )
            store.insert_triple(triple)
            n += 1
    store.finish_run(run_id)
    typer.echo(f"loaded {n} stub prediction(s) into run {run_id!r}")


if __name__ == "__main__":
    app()
