"""Extract + resolve a chosen set of documents into one run (for the website).
Usage: python run_docs.py <run_id> <scope>   scope = edgar | edgar+arxivN (N new arXiv papers)
The 20 gold documents are always included; their chunks replay from the LLM
cache (same model/prompt/repeat 0), so they cost nothing. Safe to rerun: the
run's rows are cleared first and cached chunks replay for free.
Budget guard: aborts after 3 consecutive documents whose every chunk failed."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from c2k import extract as extract_mod
from c2k import llm
from c2k import resolve as resolve_pipeline
from c2k.cli import _canonical_dir
from c2k.config import apply_arm, load_config
from c2k.store import Store

RUN_ID, SCOPE = sys.argv[1], sys.argv[2]
config = apply_arm(load_config(), "a2")
store = Store(config.paths.db, canonical_dir=_canonical_dir(config))
store.initialize()
llm.configure(store, config)

gold = sorted({r["doc_id"] for r in store._conn.execute("SELECT DISTINCT doc_id FROM gold_triples")})
others = sorted(r["doc_id"] for r in store.list_documents() if r["doc_id"] not in set(gold))
docs = list(gold) + [d for d in others if d.startswith("edgar")]
if "arxiv" in SCOPE:
    n = int(SCOPE.split("arxiv")[1])
    docs += [d for d in others if d.startswith("arxiv")][:n]
print(f"[{time.strftime('%H:%M:%S')}] {RUN_ID}: {len(docs)} docs ({len(gold)} gold, cached) model={config.model.name}", flush=True)

for table in ("triples", "mentions", "entities"):
    store._conn.execute(f"DELETE FROM {table} WHERE run_id = ?", (RUN_ID,))
store._conn.commit()
store.start_run(RUN_ID, "a2", 0, config.model.name, config.model_dump_json())

tot = dict(n_chunks=0, n_extracted=0, n_stored=0, n_span_localization_failures=0, n_deduplicated=0, n_unmapped_relations=0, n_llm_failures=0)
dead_streak = 0
t0 = time.monotonic()
for i, doc_id in enumerate(docs, 1):
    s = extract_mod.extract_document(store, config, RUN_ID, doc_id)
    for k in tot:
        tot[k] += getattr(s, k)
    dead_streak = dead_streak + 1 if (s.n_chunks and s.n_llm_failures == s.n_chunks) else 0
    print(f"[{time.strftime('%H:%M:%S')}] doc {i}/{len(docs)} {doc_id}: {s.n_chunks} chunks, {s.n_stored} stored, {s.n_llm_failures} LLM failures", flush=True)
    if dead_streak >= 3:
        print("ABORT: 3 consecutive documents failed on every chunk (credit exhausted or API down?)", flush=True)
        sys.exit(2)
print(f"[{time.strftime('%H:%M:%S')}] extraction done in {time.monotonic()-t0:.0f}s: {tot}", flush=True)
store.record_extraction_stats(RUN_ID, **tot)
r = resolve_pipeline.resolve_run(store, config, RUN_ID)
print(f"[{time.strftime('%H:%M:%S')}] resolution: {r.n_mentions} mentions -> {r.n_entities} entities ({r.n_deferred} deferred)", flush=True)
store.record_resolution_stats(RUN_ID, n_mentions=r.n_mentions, n_candidate_pairs=r.n_candidate_pairs, n_accepted_direct=r.n_accepted_direct,
    n_rejected=r.n_rejected, n_deferred=r.n_deferred, n_adjudicated_same=r.n_adjudicated_same, n_entities=r.n_entities)
store.finish_run(RUN_ID)
print(f"[{time.strftime('%H:%M:%S')}] RUN COMPLETE ({RUN_ID})", flush=True)
store.close()
