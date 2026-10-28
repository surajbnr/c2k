# C2K

Schema-constrained knowledge extraction from PDFs, with every extracted fact
traceable to an exact character span in its source document and scored
against a hand-annotated gold set.

Course project. One developer, six weeks.

## Status (2026-09-09)

| Piece | State |
|---|---|
| `store`, `config`, `schema`, `ingest`, `chunk`, `llm` | Built, tested |
| `eval/` (matching, metrics, entailment, report) | Built, tested, and **actually wired up** — every metric is either measured or honestly `n/a`, never a fabricated number (see below) |
| `extract.py`, `resolve/*`, `query.py` | Built, tested against fakes, **and now live-verified against a real LLM** (see arm A3 below) |
| `c2k ablate`, `c2k tune` | Built, tested — orchestration is real; one arm has a real sweep result now, see below |
| `scripts/fetch_corpus.py` | Built, live-verified against arXiv + SEC EDGAR |
| `relations.yaml` frozen | **Done.** Reviewed against a real 40-document sample, tagged `d5-ontology-frozen` — see "Design decisions" |
| 20-doc gold set | **Done.** 136 hand-reviewed facts (115 in-schema, 21 out-of-schema) across 20 real documents (10 arXiv + 10 EDGAR), split 5 tune / 15 test — see "Design decisions" |
| Arm A3 (local model) real run | **Done, repeat 0.** `ollama`/`qwen2.5:7b`, full extract + resolve + eval over all 20 gold docs — see below |
| Arms A0 and A2 (Claude Haiku 4.5) real run | **Done, 2 repeats each, ~$5.50 total** — see "A0 vs A2, for real" below. A1 not run; repeat 2 not run |
| The full 120-PDF corpus | **Not started at scale.** `data/pdfs/` has the 40-doc sample used for the D5 and D24 review passes (real, kept — a head start, not thrown away) |

In short: **the code is done and its own measurement is trustworthy, the
ontology is frozen and reviewed against real documents, a real 20-document
gold set exists, and one ablation arm has genuine (if rough) numbers behind
it** — but three of the four arms and the full 120-document corpus remain,
both blocked on things unavailable here (API budget; the rest of the corpus).

**Arm A3, for real, 2026-09-09:** with no funded Anthropic API key, arm A3
(`ollama`/`qwen2.5:7b`, fully local and free) ran end-to-end over all 20
gold-annotated documents — genuinely the first non-fake, non-mocked run this
project has done. Extraction: 203 chunks → 351 triples stored (2h54m).
Resolution: 702 mentions → 422 entities (19min). Real eval on the test
split: triple F1 1.9% (loose), ER pairwise F1 93.9%. The low triple F1 isn't
a pipeline bug — watching it run surfaced exactly why: a 7B local model
struggles badly at staying inside a closed relation vocabulary and at
quoting evidence verbatim (see "Design decisions" for the specific failure
modes observed). Running this for real also found and fixed two genuine
robustness bugs — a single failed LLM call used to abort an entire
multi-hour corpus batch with zero progress saved; a fake-provider test would
never have caught this, since a fake provider doesn't fail the way a real
small model does under load.

**A0 vs A2, for real, 2026-09-27 and 2026-10-02** (`claude-haiku-4-5`, 2
repeats each, the 20 gold documents, scored on the 15-document test split,
~$5.50 of API spend):

| | A0 (unconstrained) | A2 (constrained + resolution) | A3 (local qwen2.5:7b, 1 repeat) |
|---|---|---|---|
| triple F1 (loose), repeats | 6.3%, 6.2% | 7.3%, 7.1% | 2.6% |
| triple recall (loose) | 30.0%, 30.0% | 36.2%, 35.0% | 5.0% |
| triple precision (loose) | 3.5%, 3.5% | 4.1%, 3.9% | 1.8% |
| span-localization failures | 4.6%, 2.7% | 2.3%, 2.9% | 19.4% |
| ER pairwise F1 | n/a | 91.7%, 96.8% | 93.9% |
| hallucination, LLM judge (repeat 0 only, n=100) | 25% | 21% | not run |

Read this carefully. By the letter of the F1 half of the success criterion,
A2 passes: the mean gap (~0.95 points) is larger than either arm's
repeat-to-repeat range (0.1-0.2 points). But that range is small partly
by construction (temperature is 0, so repeats are nearly identical), and it
says nothing about the bigger uncertainty: the gold set has 80 in-schema
test facts, so a 1-point F1 gap is about 5 facts. Only 2 of the 3 specified
repeats were run. The hallucination half is **not evaluated**: it needs the
human-labelled rate (`c2k judge`, interactive), which has not been done;
the LLM-judge figures above are repeat 0 only. Precision is low for every
arm because the gold set is sparse (a correct but unannotated triple counts
as a false positive). This comparison also exposed and fixed an evaluator
bug (predictions from tune-split documents were counted in the test-split
precision denominator), and `anthropic` is pinned `<1` because 1.x removed
the `temperature` argument the project requires.

**What changed in the most recent pass:** the evaluator used to silently
under-report. `c2k eval` never received span-localization-failure counts,
adjudication counts, entailment verdicts, or entity-resolution clusters — so
it printed `n/a` for most metrics, and for one (span-localization failure
rate) it printed a **fabricated `0.0%`** instead of `n/a`, because `0` was the
silent default rather than "not measured." All of that is now persisted (a
new `run_stats` table, written incrementally by `extract`/`resolve`/`ablate`)
and read back automatically; A0's unmapped free-form predicates now count
against precision, as the ablation design requires; the LLM entailment judge
and Cohen's κ are wired into `c2k judge` alongside the human labels. See
"Design decisions" below for the full account.

## Quickstart

```bash
# one-time setup
curl -LsSf https://astral.sh/uv/install.sh | sh   # installs uv to ~/.local/bin
export PATH="$HOME/.local/bin:$PATH"
uv sync                                            # provisions Python 3.11 + all deps

# run the test suite
PYTHONPATH="" uv run pytest -q
```

You should see `102 passed`. Every subsequent `uv run ...` command in this
README assumes you're in the repo root with `uv`'s bin dir on `PATH`.

### ⚠️ If pytest crashes with `ModuleNotFoundError: No module named 'lark'`

Your shell has ROS2's `PYTHONPATH` leaking in (a `setup.bash` sourced
somewhere), which makes pytest try to load ROS's `launch_testing` plugin.
Always clear it for this project:

```bash
PYTHONPATH="" uv run pytest -q
PYTHONPATH="" uv run c2k <command>
```

This has nothing to do with the project — it's specific to this machine.

## What's actually implemented

A pipeline with five real stages plus an evaluator, all wired through one
CLI (`c2k`, via [Typer](https://typer.tiangolo.com/)):

```
PDF ──ingest──▶ canonical text ──chunk──▶ chunks ──extract──▶ triples ──resolve──▶ resolved triples
                                                                                          │
                                                                                          ▼
                                                                          eval / query / ablate
```

- **`ingest`** (`c2k/ingest.py`) — PyMuPDF `page.get_text("words")`, words
  joined with single spaces, pages with `"\n\n"`, into one canonical text
  string per document. Every offset in the system indexes into this string.
  A `(char_range) -> (page, bbox)` sidecar index is also written, so a span
  can be traced back to a page location. PyMuPDF was chosen deliberately over
  Docling: Docling's markdown export reflows characters, which breaks the
  offset guarantee everything downstream relies on.
- **`chunk`** (`c2k/chunk.py`) — 900 tokens / 150 overlap (`cl100k_base`),
  boundaries snapped to the nearest sentence break within ±50 tokens.
- **`extract`** (`c2k/extract.py`) — one LLM call per chunk, returning
  subject/relation/object/evidence_text (no offsets — models can't count
  characters reliably). Post-processing locates `evidence_text` in the chunk:
  exact match → nearest occurrence to subject+object → fuzzy
  (`rapidfuzz.partial_ratio_alignment`, ≥90) → drop-and-log. Duplicate
  triples from overlapping chunks collapse, keeping the higher confidence.
- **`resolve`** (`c2k/resolve/`) — mention coreference in three passes:
  1. `blocking.py`: embed each mention (`BAAI/bge-base-en-v1.5`), FAISS
     `IndexFlatIP` top-20 candidate pairs.
  2. `filtering.py`: `rapidfuzz.token_set_ratio` on normalized surface
     strings; below the low threshold reject, above the high threshold
     accept, else defer.
  3. `adjudicate.py`: one LLM call per batch of 20 deferred pairs.
  4. `cluster.py`: union-find over accepted pairs → entities (canonical name
     = most frequent surface, ties broken by longest).
- **`eval/`** — Hungarian one-to-one matching
  (`scipy.optimize.linear_sum_assignment`, deliberately never greedy, so
  results don't depend on row order) between predicted and gold triples:
  precision/recall/F1 (both loose and span-strict), hallucination rate, ER
  pairwise F1, schema recall ceiling, span-localization failure rate,
  adjudication rate, and cost, plus human/LLM entailment judging with
  Cohen's κ. `evaluate_run` pulls everything it needs from the DB by
  default — pipeline stats from the `run_stats` table,
  entailment verdicts from `entailment_judgements`, ER clusters derived from
  resolved entities and (on the gold side) nearest-match to gold
  subject/object strings. A stage that never ran reports `n/a`, not a
  fabricated `0`.
- **`query`** (`c2k/query.py`) — loads a run's triples into a NetworkX
  `MultiDiGraph`; a small heuristic (no LLM) parses a query string into a
  relation keyword + entity guess; every returned fact is re-verified
  against the canonical text before being returned (a stale span is dropped
  and logged, never returned with a warning).

## The six invariants

These are asserted in code, not just documented:

| ID | What | Where |
|---|---|---|
| I1 | `canonical_text[start:end] == evidence_text` | `store.insert_triple` raises `AssertionError` otherwise |
| I2 | No triple without a resolvable span | `schema.Triple`'s pydantic validators — there's no constructor path around this |
| I3 | Relations only from a closed vocabulary | `schema.Relation` is built from `relations.yaml`; `tests/test_relations.py` AST-scans the whole `c2k/` tree for stray relation string literals |
| I4 | Only `llm.py` touches `anthropic`/`ollama` | `tests/test_no_direct_sdk.py` AST-scans for it |
| I5 | Test split is read-only outside `eval`/`ablate` | `store.load_gold(Split.TEST)` raises `TestSplitLocked` unless handed a `TestUnlock`, which only `cli.py`'s `eval` and `ablate` commands construct — `tests/test_split_lock.py` checks `tune`'s source never even names `TestUnlock` |
| I6 | Every run reproducible from `runs.config_json` + the LLM cache | `llm.py` caches on `sha256(model \| prompt \| schema \| temperature \| repeat_idx)` |

## The CLI

```bash
uv run c2k ingest      --input data/pdfs                 # PDFs -> canonical text
uv run c2k chunk       --doc-id all                       # canonical text -> chunks
uv run c2k extract     --arm a2 --repeat 0                # chunks -> triples (needs ANTHROPIC_API_KEY)
uv run c2k resolve     --run-id RUN                        # mention coreference cascade
uv run c2k eval        --run-id RUN --split tune|test      # metrics table
uv run c2k judge       --run-id RUN --n 100                # human + LLM entailment labelling (interactive; needs ANTHROPIC_API_KEY for the LLM half)
uv run c2k ablate      --arms a0,a1,a2,a3 --repeats 3      # full sweep (needs ANTHROPIC_API_KEY)
uv run c2k tune        --split tune --run-id RUN           # tau_lo/tau_hi grid sweep
uv run c2k query       "which companies acquired X" --run-id RUN
uv run c2k annotate    --doc-id DOC                        # append gold facts (interactive)
uv run c2k stub-run    --from FILE --run-id RUN            # load fake predictions (no LLM needed)
```

`query` and `tune` both take a `--run-id` — every command that reads triples
requires one, so these do too, even though it's easy to forget. Documented
inline in `c2k/cli.py`.

**The four ablation arms** (`c2k.config.ARM_TABLE`):

| Arm | constrained? | resolution? | model | isolates |
|---|---|---|---|---|
| A0 | no | no | claude | unconstrained baseline |
| A1 | yes | no | claude | constrained decoding |
| A2 | yes | yes | claude | full system |
| A3 | yes | yes | qwen2.5:7b (ollama) | model capability vs. constraint |

`arm` is a config value, never a code branch — `apply_arm()` in `config.py`
derives the per-arm `Config` from `config/default.yaml`.

## Building the corpus

`data/pdfs/` is empty and the pipeline assumes ~120 PDFs already exist.
`scripts/fetch_corpus.py` (a standalone script, not a `c2k` subcommand) fills
it from two public sources, chosen to exercise both halves of the mixed
corporate/scientific relation vocabulary:

```bash
# ~60 recent arXiv cs.CL/cs.LG papers -> authored_by, evaluated_on, uses_method, ...
uv run python3 scripts/fetch_corpus.py --source arxiv --n 60 --out data/pdfs

# ~60 SEC EDGAR 8-K filings from 30 large-cap companies -> acquired, ceo_of, ...
# SEC requires a reachable contact in the User-Agent or it 403s.
uv run python3 scripts/fetch_corpus.py --source edgar --n 60 --out data/pdfs \
    --contact you@example.com

# see what would be fetched without downloading anything
uv run python3 scripts/fetch_corpus.py --source arxiv --n 5 --out data/pdfs --dry-run
```

EDGAR filings are HTML; they're rendered to PDF with `pymupdf.Story` (already
a dependency — nothing new to install). Both sources write to
`data/pdfs/MANIFEST.jsonl` (source URL + fetch date per file), are idempotent
(re-running skips files already on disk), and were live-verified against the
real APIs — a fetched arXiv paper and a fetched 8-K both round-tripped
correctly through `c2k ingest`/`c2k chunk` during development.

Once fetched: `uv run c2k ingest --input data/pdfs && uv run c2k chunk --doc-id all`,
then hand-annotate 20 of them (`uv run c2k annotate --doc-id X`) into
`data/gold/`.

## Running the demo (no PDFs, no API key)

There's no real corpus yet, so `tests/fixtures/` has a hand-built demo
document (7 predictions, 7 gold facts) that exercises `stub-run` → `eval`
end-to-end. This is the same thing `pytest` checks, run manually:

```bash
# seed a demo document + gold facts into data/db/c2k.sqlite
uv run python3 - <<'EOF'
from pathlib import Path
import shutil
from c2k.config import load_config
from c2k.cli import _canonical_dir
from c2k.store import Store

config = load_config()
store = Store(config.paths.db, canonical_dir=_canonical_dir(config))
store.initialize()

text = Path("tests/fixtures/demo001_canonical.txt").read_text()
store.insert_document("demo001", "tests/fixtures/demo001_canonical.txt", "n/a", len(text), "tune")
store.save_canonical_text("demo001", text)

Path(config.paths.gold).mkdir(parents=True, exist_ok=True)
shutil.copy("tests/fixtures/demo001_gold.jsonl", Path(config.paths.gold) / "demo001.jsonl")
store.sync_gold_from_jsonl(config.paths.gold)
EOF

# load 7 stub predictions as a real run (no LLM call needed — this is what
# "eval against stubs" means before extraction is run for real)
PYTHONPATH="" uv run c2k stub-run --from tests/fixtures/stub_predictions.jsonl --run-id demo

# print the metrics table — span-localization-failure-rate and
# adjudication-rate will read "n/a" here: this run never went through
# extract/resolve, so there's genuinely nothing to report for them yet
PYTHONPATH="" uv run c2k eval --run-id demo --split tune

# real resolution cascade — downloads BAAI/bge-base-en-v1.5 on first run
PYTHONPATH="" uv run c2k resolve --run-id demo

# re-run eval: ER pairwise F1 and adjudication rate are now real numbers,
# read back automatically from the DB (no flags needed)
PYTHONPATH="" uv run c2k eval --run-id demo --split tune

# explore the graph in a notebook
jupyter notebook notebooks/query_demo.ipynb
# or query from the CLI directly:
PYTHONPATH="" uv run c2k query "who acquired Mentor Graphics" --run-id demo
```

Inspect the raw DB (no `sqlite3` CLI on this machine — use Python):

```bash
uv run python3 -c "
import sqlite3
conn = sqlite3.connect('data/db/c2k.sqlite')
conn.row_factory = sqlite3.Row
for r in conn.execute('SELECT triple_id, relation, evidence_text FROM triples'):
    print(dict(r))
"
```

### Running it on your own PDFs

```bash
cp your_files/*.pdf data/pdfs/
PYTHONPATH="" uv run c2k ingest --input data/pdfs
PYTHONPATH="" uv run c2k chunk --doc-id all
export ANTHROPIC_API_KEY=sk-...
PYTHONPATH="" uv run c2k extract --arm a2 --repeat 0
PYTHONPATH="" uv run c2k resolve --run-id a2_r0
```

`c2k extract`/`ablate` are the only commands that call a real LLM
(`c2k/llm.py` is the sole module allowed to import `anthropic`/`ollama`).
`resolve` calls an LLM too, but only for pairs that land in the ambiguous
similarity band between the two resolution thresholds — most corpora will
rarely hit it.

## Repository layout

```
c2k/
  config.py          # pydantic-settings config, arm overrides (ARM_TABLE)
  schema.py           # Relation enum (from relations.yaml), Triple/Mention/GoldTriple
  llm.py              # the only module importing anthropic/ollama; caches every call
  norm.py             # shared string normalization (used by resolve + eval)
  ingest.py           # PDF -> canonical text + page/bbox offset index
  chunk.py            # canonical text -> overlapping, sentence-snapped chunks
  extract.py          # chunk -> triples, with span localization + A0 relation mapping
  resolve/
    blocking.py       # embeddings + FAISS -> candidate pairs
    filtering.py       # rapidfuzz -> accept / reject / defer
    adjudicate.py      # batched LLM calls on the deferred band
    cluster.py         # union-find -> entities
    __init__.py         # orchestrates the above; compute_resolution() is DB-free
  store.py            # all SQLite access goes through here; I1/I5 enforced here
  query.py            # NetworkX traversal + span re-verification
  eval/
    match.py           # Hungarian matching, loose + span-strict
    metrics.py          # every metric in the results table
    entailment.py        # human + LLM entailment judging, Cohen's kappa
    report.py            # text table rendering, success-criterion check
  cli.py              # every CLI command
config/
  default.yaml        # the one config file everything reads
  relations.yaml       # the closed relation vocabulary (14 relations)
  splits.yaml          # doc_id -> tune/test assignment (empty until gold exists)
prompts/
  extract.md, adjudicate.md, entail.md
data/
  pdfs/, gold/, db/, canonical/   # all empty/gitignored except .gitkeep
scripts/
  fetch_corpus.py     # arXiv + SEC EDGAR corpus assembly (not a c2k subcommand)
tests/                # 102 tests; fixtures/ has the hand-built demo doc + corpus-fetcher fixtures
notebooks/
  query_demo.ipynb    # a notebook is the interface — no web UI in this project
```

## Configuration

Everything reads `config/default.yaml`. Point at a different file with the
`C2K_CONFIG` environment variable (useful for an isolated scratch run without
touching the real `data/` directories):

```bash
C2K_CONFIG=/path/to/scratch-config.yaml uv run c2k eval --run-id RUN --split tune
```

Key knobs: `chunking.{tokens,overlap,snap_window}`,
`resolution.{tau_lo,tau_hi,top_k,adjudication_batch}`,
`evaluation.{entity_match_threshold,span_iou_threshold,repeats}`.
`resolution.tau_lo`/`tau_hi` are tuned defaults — `c2k tune` finds better
ones but never writes them back; freezing a new value in the YAML is a
human decision.

## Testing

```bash
PYTHONPATH="" uv run pytest -q              # all 102 tests
PYTHONPATH="" uv run pytest -q tests/test_metrics.py   # just the evaluator
PYTHONPATH="" uv run pytest -q tests/test_fetch_corpus.py   # corpus fetcher, offline only
```

Every test involving an LLM call registers a fake provider in
`llm._PROVIDERS["fake"]` (see `tests/test_cache.py` for the pattern) — no
test in this repo makes a real network call to Anthropic or Ollama.
`tests/test_resolve.py` similarly monkeypatches the embedding step, since
loading `sentence-transformers` needs a one-time model download.

## Design decisions

A running log of choices made where the original design left a gap, most
recent first:

- **Arm A3 run for real via a local `ollama` install, no API budget needed**:
  with no funded Anthropic key, set up `ollama` as a portable binary in
  `~/.local/ollama` (no sudo — this machine has none) and pulled
  `qwen2.5:7b`, then ran extract + resolve + eval for arm A3 over all 20
  gold-annotated documents. Two real robustness bugs surfaced and got fixed
  as a direct result — a single LLM call failing after retries (which
  `llm.call()` correctly raises on, by design) had no handler one level up
  in either `extract.py` or `c2k/resolve/adjudicate.py`, so it silently
  discarded an entire multi-hundred-item batch's prior progress. Both now
  catch per-item, log, and continue (new `ExtractionStats.n_llm_failures`;
  `adjudicate_pairs` treats a failed batch's pairs as conservative "not the
  same entity" rather than raising). Small-model failure modes observed
  live, worth knowing when interpreting A3's eventual numbers: emitting
  relations outside the closed enum despite constrained decoding; echoing
  the extraction prompt's own few-shot exemplar sentences back as if
  extracted from the current chunk; literal placeholder strings like `"not
  explicitly mentioned in the chunk"` in the evidence field instead of
  declining to extract; confusing the pydantic schema definition itself
  with actual field values in a tool call; not reliably returning an empty
  `triples: []` when nothing in a chunk matches. None of this is a pipeline
  bug — it's exactly the "model capability vs constraint" comparison arm A3
  exists to make, just observed directly instead of only inferred from
  aggregate metrics.
- **20-document gold set, AI-drafted and human-reviewed, not auto-generated
  wholesale**: for each of the 20 documents, facts were drafted by reading the
  real canonical text and computing exact character-offset evidence spans
  (verified against `gold_integrity_report` before being written), presented
  for review, then committed — never written straight to `data/gold/` without
  that step. The annotator of record is the user, consistent with the spec's
  single-annotator design; this is meant as an assistive drafting workflow,
  not a substitute for it. Result: 136 facts (115 in-schema, 21 out-of-schema)
  across 20 documents (10 arXiv, 10 EDGAR), split 5 tune / 15 test in
  `config/splits.yaml`. Confirms the D5 review's finding in practice rather
  than just in theory: the 10 EDGAR 8-Ks average ~2 facts each (most are
  routine quarterly-earnings filings with nothing beyond a `headquartered_in`
  address block), while the 10 arXiv papers average ~11 each — well short of
  the spec's aspirational ~20/document average, but an honest count rather
  than a padded one. Relation coverage lines up exactly with the D5
  prediction: `subsidiary_of`, `ceo_of`, `founded_by`, `partnered_with`, and
  `invested_in` have zero gold facts, for the same reason `relations.yaml`'s
  header already documents. Two annotation judgment calls worth knowing
  about: a paper with no institution string in its byline (only an email
  domain) got no `affiliated_with` fact rather than inferring one from the
  domain, and a paper that explicitly hedges "do not show that knowledge
  graphs universally outperform natural-language notes" got no `outperforms`
  fact rather than one that would misstate its actual claim.
- **Ontology frozen against real documents, not guesswork**: before tagging
  `config/relations.yaml` as final, fetched a 40-document validation sample
  (15 arXiv cs.CL/cs.LG papers + 25 SEC EDGAR 8-Ks) and scanned the real
  canonical text for each of the 14 relations' phrasing. 7 relations
  (`acquired`, `headquartered_in`, `evaluated_on`, `outperforms`,
  `reports_metric`, `funded_by`, `uses_method`) had strong real support —
  one exemplar each was swapped for a real verbatim sentence from the
  sample. 5 (`subsidiary_of`, `ceo_of`, `founded_by`, `partnered_with`,
  `invested_in`) had **zero** real hits — not because they're the wrong
  relations, but because an SEC 8-K's primary document only covers one
  reportable event, and this sample happened to draw mostly routine
  quarterly-earnings filings. Real coverage for those five will need
  different EDGAR document types (DEF 14A proxies, 10-K Exhibit 21,
  event-specific 8-Ks) that the fetcher below doesn't target yet — flagged
  inline in `config/relations.yaml` and worth addressing before annotating
  gold facts against them. All 14 relations were kept; this was a review
  pass, not a rewrite. Tagged `d5-ontology-frozen`.
- **Corpus fetcher (`scripts/fetch_corpus.py`)**: nothing originally
  specified how the ~120-PDF corpus gets acquired — annotation assumes the
  PDFs already exist. Added a standalone script (not a `c2k` subcommand, so
  the CLI surface stays exactly what's documented above) that pulls ~60
  arXiv cs.CL/cs.LG papers and ~60 SEC EDGAR filings (rendered to PDF via
  `pymupdf.Story` — already a dependency, no new one added) into
  `data/pdfs/`, with a `MANIFEST.jsonl` recording source URL and fetch date
  per file.
- **`run_stats` table**: extraction/resolution stats used to exist only as
  in-memory objects the CLI printed and discarded, so `c2k eval` run in a
  later process had no span-localization-failure count, candidate-pair
  count, or adjudicated-pair count to report — it was silently reporting a
  fabricated `0.0%` for span-localization failure rate instead of `n/a`.
  Added a `run_stats` table (keyed by `run_id`, written incrementally by
  `extract`/`resolve`/`ablate`) rather than columns on the existing `runs`
  table, so that table's schema stays untouched. `c2k eval`'s metrics now
  default to loading from this table instead of `0`; an explicit argument
  (including `0`) still overrides it, which is what keeps the evaluator's
  own unit tests store-free.
- **Gold-derived ER clusters**: ER pairwise F1 needs a gold "same-entity"
  clustering, but the gold JSONL format records triples, not coreference.
  Derived it post-hoc: each mention is assigned to whichever gold
  subject/object string in the same document it's closest to (by the same
  normalize+similarity rule used for triple matching), at the configured
  entity-match threshold; a mention matching nothing above threshold
  contributes no pairs. This is the only derivation the gold format admits —
  worth revisiting if a future annotation workflow should record
  coreference directly instead.
- **Unmapped A0 predicates in precision**: A0's unmapped free-form
  predicates are supposed to "count as misses," but an unmapped predicate
  structurally cannot become a stored `Triple` (the relation field is
  constrained to the closed enum), so it never reached the evaluator's
  predicted list at all. `triple_prf1` now takes `n_unmatchable_predicted`
  and folds it into the precision denominator only (recall — gold coverage —
  is unaffected by a prediction that could never have matched either way).
- **Canonical text storage location**: canonical text lives at
  `data/canonical/{doc_id}.txt`, a sibling of `pdfs/`/`gold/`/`db/` under
  the same data root (`cli._canonical_dir()`).
- **Gold JSONL → DB sync**: gold facts live in `data/gold/{doc_id}.jsonl`
  as the durable source of truth, but `store.load_gold` reads a
  `gold_triples` SQLite table. `c2k ingest` calls
  `store.sync_gold_from_jsonl(config.paths.gold)` after ingesting PDFs, so
  re-running ingest picks up new/edited gold files.
- **`c2k stub-run`**: loads a JSONL of predictions as a real run with zero
  PDFs and no API key, so the evaluator is demonstrable end-to-end before
  extraction is run for real.
- **`--run-id` on `tune` and `query`**: every other command that reads
  triples requires one — added for consistency.

## What's next

1. ~~Freeze `config/relations.yaml` and tag the commit.~~ **Done** — see
   "Design decisions" above.
2. ~~Hand-annotate a 20-document gold set.~~ **Done** — 136 facts, split 5
   tune / 15 test, `tests/test_gold_integrity.py` passes. If
   `subsidiary_of`/`ceo_of`/`founded_by`/`partnered_with`/`invested_in`
   should have real coverage before the full corpus is annotated,
   `scripts/fetch_corpus.py` needs to target different EDGAR document types
   first (DEF 14A proxies, 10-K Exhibit 21, event-specific 8-Ks) — not
   attempted yet; a live test of DEF 14A fetching during this pass hung at
   6.5GB RAM rendering one proxy statement via `pymupdf.Story` and had to be
   killed, so that path needs its own fix, not just a new `--filing-type`
   flag.
3. ~~Run `scripts/fetch_corpus.py` for both sources at full scale.~~
   **Done** — `data/pdfs/` now holds 120 PDFs (60 arXiv + 60 EDGAR 8-K,
   listed in `MANIFEST.jsonl`). EDGAR uses `--max-per-cik 3` so one company
   can't dominate (the original 40-doc sample already had 15 Apple filings,
   which stay). Still to do: hand-annotate the remaining 100 documents into
   `data/gold/`. The new PDFs are not yet ingested (`c2k ingest`), and
   evaluation numbers don't change until they carry gold facts.
4. ~~Get one arm running for real, end to end.~~ **Done** — arm A3
   (`ollama`/`qwen2.5:7b`) ran extract + resolve + eval over all 20
   gold-annotated documents; see "Arm A3, for real" above. Repeats 1-2 for
   A3 are in progress. Note they run with a per-call cap (2048 output
   tokens, 5-minute timeout, local provider only), which repeat 0 did not
   have: during repeat 1 one call looped endlessly for over an hour until
   the GPU runner crashed. Well-behaved calls are unaffected; a runaway one
   would fail validation either way, just faster.
5. `c2k ablate --arms a2 --repeats 1` on the real corpus — still blocked on
   a funded `ANTHROPIC_API_KEY` (the key on file is valid, the account has
   no credit balance).
6. All 4 arms × 3 repeats; read off the success criterion the `ablate`
   command already prints (A2 must beat A0 on triple F1 by more than the
   3-repeat range, with hallucination no worse — reported honestly either
   way, including as a negative result).
