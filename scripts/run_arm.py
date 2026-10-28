"""Run one arm+repeat over the gold-annotated docs only (same as run_a3_repeat.py, any arm).
Usage: python run_arm.py <arm> <repeat>   e.g. a2 0. Evaluate afterwards with `c2k eval`.
Exits with status 2 (run left unfinished) if a paid-API run loses more than 5% of its sections."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from c2k import extract as extract_mod
from c2k import llm
from c2k import resolve as resolve_pipeline
from c2k.cli import _canonical_dir
from c2k.config import apply_arm, load_config
from c2k.store import Store

ARM, REPEAT = sys.argv[1], int(sys.argv[2])
run_id = f"{ARM}_r{REPEAT}"

config = apply_arm(load_config(), ARM)
store = Store(config.paths.db, canonical_dir=_canonical_dir(config))
store.initialize()
llm.configure(store, config)

gold_docs = sorted({r["doc_id"] for r in store._conn.execute("SELECT DISTINCT doc_id FROM gold_triples").fetchall()})
print(f"[{time.strftime('%H:%M:%S')}] {run_id}: model={config.model.provider}/{config.model.name} "
      f"constrained={config.extraction.constrained} resolution={config.resolution.enabled} docs={len(gold_docs)}", flush=True)
store.start_run(run_id, ARM, REPEAT, config.model.name, config.model_dump_json())

t0 = time.monotonic()
stats = extract_mod.extract_corpus(store, config, run_id, doc_ids=gold_docs)
print(f"[{time.strftime('%H:%M:%S')}] extraction done in {time.monotonic()-t0:.0f}s: {stats.n_chunks} chunks -> "
      f"{stats.n_extracted} extracted, {stats.n_stored} stored, {stats.n_span_localization_failures} span failures, "
      f"{stats.n_deduplicated} deduped, {stats.n_unmapped_relations} unmapped, {stats.n_llm_failures} LLM failures", flush=True)
# A paid API that fails on many sections means a bad key, an empty balance or an outage.
# Stop without marking the run finished, so it cannot be mistaken for a complete one.
# (The local model is excluded: it legitimately fails a few percent of sections.)
if config.model.provider == "anthropic" and stats.n_chunks and stats.n_llm_failures > max(5, 0.05 * stats.n_chunks):
    print(f"ABORT: {stats.n_llm_failures} of {stats.n_chunks} sections failed (bad key, empty balance or API down?). "
          "Run left unfinished.", flush=True)
    sys.exit(2)
store.record_extraction_stats(
    run_id, n_chunks=stats.n_chunks, n_extracted=stats.n_extracted, n_stored=stats.n_stored,
    n_span_localization_failures=stats.n_span_localization_failures, n_deduplicated=stats.n_deduplicated,
    n_unmapped_relations=stats.n_unmapped_relations, n_llm_failures=stats.n_llm_failures,
)
if config.resolution.enabled:
    r = resolve_pipeline.resolve_run(store, config, run_id)
    print(f"[{time.strftime('%H:%M:%S')}] resolution: {r.n_mentions} mentions -> {r.n_entities} entities "
          f"({r.n_deferred} deferred, {r.n_adjudicated_same} same)", flush=True)
    store.record_resolution_stats(
        run_id, n_mentions=r.n_mentions, n_candidate_pairs=r.n_candidate_pairs, n_accepted_direct=r.n_accepted_direct,
        n_rejected=r.n_rejected, n_deferred=r.n_deferred, n_adjudicated_same=r.n_adjudicated_same, n_entities=r.n_entities,
    )
store.finish_run(run_id)
print(f"[{time.strftime('%H:%M:%S')}] RUN COMPLETE ({run_id})", flush=True)
store.close()
