"""CLI-level wiring tests: `c2k extract`/`resolve`/`query`/`ablate` invoked
through the real Typer app (not just the underlying module functions), to
catch config/path/run_id wiring bugs that per-module unit tests can't see —
this is exactly the class of bug (`_canonical_dir` never being passed to
`Store`) that a live manual run caught earlier in this project.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from c2k import llm
from c2k.cli import _canonical_dir, _store_from_config, app
from c2k.config import load_config
from c2k.extract import ExtractionResponse
from c2k.resolve.adjudicate import BatchVerdicts
from c2k.schema import ExtractedTriple, Relation, Triple
from c2k.store import Store

runner = CliRunner()


@pytest.fixture
def scratch_env(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "arm": "a1",
                "model": {"provider": "fake", "name": "fake-model", "temperature": 0.0},
                "chunking": {"tokens": 900, "overlap": 150, "snap_window": 50},
                "extraction": {"constrained": True, "max_retries": 3, "min_confidence": 0.0},
                "resolution": {
                    "enabled": True,
                    "embedding_model": "fake-embedder",
                    "top_k": 20,
                    "tau_lo": 0.72,
                    "tau_hi": 0.88,
                    "adjudication_batch": 20,
                },
                "evaluation": {
                    "entity_match_threshold": 0.85,
                    "span_iou_threshold": 0.5,
                    "repeats": 2,
                    "judge_sample": 100,
                },
                "paths": {
                    "pdfs": str(tmp_path / "pdfs"),
                    "gold": str(tmp_path / "gold"),
                    "db": str(tmp_path / "db" / "c2k.sqlite"),
                },
            }
        )
    )
    monkeypatch.setenv("C2K_CONFIG", str(config_path))
    return tmp_path


@pytest.fixture
def fake_provider(monkeypatch):
    queue: list = []

    def _provider(prompt, response_model, *, model, temperature, max_retries):
        assert queue, "test queued no fake response"
        return queue.pop(0), {"input_tokens": 1, "output_tokens": 1}

    monkeypatch.setitem(llm._PROVIDERS, "fake", _provider)
    return queue


def _seed_document_and_chunk(scratch_env: Path, text: str, doc_id: str = "doc1") -> None:
    config = load_config()
    store = Store(config.paths.db, canonical_dir=_canonical_dir(config))
    store.initialize()
    store.insert_document(doc_id, f"{doc_id}.pdf", "sha", len(text), "train_unused")
    store.save_canonical_text(doc_id, text)
    store.insert_chunk(f"{doc_id}:0", doc_id, 0, 0, len(text), text)
    store.close()


def test_extract_command_populates_db(scratch_env: Path, fake_provider) -> None:
    text = "Siemens acquired Mentor Graphics in 2017."
    _seed_document_and_chunk(scratch_env, text)

    fake_provider.append(
        ExtractionResponse(
            triples=[
                ExtractedTriple(
                    subject="Siemens",
                    relation=Relation.acquired,
                    object="Mentor Graphics",
                    evidence_text="Siemens acquired Mentor Graphics",
                    confidence=0.9,
                )
            ]
        )
    )

    result = runner.invoke(app, ["extract", "--arm", "a1", "--repeat", "0"])
    assert result.exit_code == 0, result.output
    assert "1 stored" in result.output

    config = load_config()
    store = Store(config.paths.db, canonical_dir=_canonical_dir(config))
    triples = store.list_triples("a1_r0")
    stats = store.get_run_stats("a1_r0")
    store.close()
    assert len(triples) == 1
    assert triples[0]["relation"] == "acquired"
    # `extract` must persist ExtractionStats to `run_stats`, not just print
    # it — otherwise a later `c2k eval` in a fresh process has nothing to
    # compute span-localization-failure-rate against.
    assert stats is not None
    assert stats["n_stored"] == 1
    assert stats["n_span_localization_failures"] == 0


def test_judge_command_records_human_and_llm_verdicts(scratch_env: Path, fake_provider) -> None:
    from c2k.eval.entailment import EntailmentVerdict

    text = "Siemens acquired Mentor Graphics in 2017."
    _seed_document_and_chunk(scratch_env, text)
    fake_provider.append(
        ExtractionResponse(
            triples=[
                ExtractedTriple(
                    subject="Siemens",
                    relation=Relation.acquired,
                    object="Mentor Graphics",
                    evidence_text="Siemens acquired Mentor Graphics",
                    confidence=0.9,
                )
            ]
        )
    )
    extract_result = runner.invoke(app, ["extract", "--arm", "a1", "--repeat", "0"])
    assert extract_result.exit_code == 0, extract_result.output

    fake_provider.append(EntailmentVerdict(entails=True, reason="matches"))
    result = runner.invoke(app, ["judge", "--run-id", "a1_r0", "--n", "1"], input="y\n")
    assert result.exit_code == 0, result.output
    assert "human=0.0" in result.output
    assert "llm=0.0" in result.output
    assert "kappa=1.0" in result.output

    config = load_config()
    store = Store(config.paths.db, canonical_dir=_canonical_dir(config))
    human = store.get_entailment_judgements("a1_r0", "human")
    llm_judgements = store.get_entailment_judgements("a1_r0", "llm")
    store.close()
    assert len(human) == 1 and human[0]["verdict"] == 1
    assert len(llm_judgements) == 1 and llm_judgements[0]["verdict"] == 1


def test_query_command_after_extract(scratch_env: Path, fake_provider) -> None:
    text = "Siemens acquired Mentor Graphics in 2017."
    _seed_document_and_chunk(scratch_env, text)
    fake_provider.append(
        ExtractionResponse(
            triples=[
                ExtractedTriple(
                    subject="Siemens",
                    relation=Relation.acquired,
                    object="Mentor Graphics",
                    evidence_text="Siemens acquired Mentor Graphics",
                    confidence=0.9,
                )
            ]
        )
    )
    extract_result = runner.invoke(app, ["extract", "--arm", "a1", "--repeat", "0"])
    assert extract_result.exit_code == 0, extract_result.output

    query_result = runner.invoke(app, ["query", "who acquired Mentor Graphics", "--run-id", "a1_r0"])
    assert query_result.exit_code == 0, query_result.output
    assert "Siemens" in query_result.output
    assert "Mentor Graphics" in query_result.output


def test_resolve_command_persists_run_stats(scratch_env: Path, monkeypatch) -> None:
    import numpy as np

    from c2k.resolve import blocking

    config = load_config()
    store = Store(config.paths.db, canonical_dir=_canonical_dir(config))
    store.initialize()
    store.insert_document("doc1", "doc1.pdf", "sha", 20, "tune")
    store.save_canonical_text("doc1", "Siemens AG and Siemens Corp merged.")
    store.start_run("run1", "a2", 0, "fake-model", "{}")
    store.insert_mention("m1", "run1", "doc1", "Siemens AG", 0, 10)
    store.insert_mention("m2", "run1", "doc1", "Siemens Corp", 15, 27)
    store.finish_run("run1")
    store.close()

    # Two mentions, identical embedding -> similarity 1.0 -> ACCEPT band
    # directly, no LLM adjudication needed (keeps this test network-free).
    monkeypatch.setattr(
        blocking, "embed_mentions", lambda mentions, canonical_by_doc, *, model_name: np.ones((len(mentions), 4), dtype="float32")
    )

    result = runner.invoke(app, ["resolve", "--run-id", "run1"])
    assert result.exit_code == 0, result.output

    store = Store(config.paths.db, canonical_dir=_canonical_dir(config))
    stats = store.get_run_stats("run1")
    store.close()
    assert stats is not None
    assert stats["n_mentions"] == 2
    assert stats["n_entities"] == 1  # merged into one entity


def test_tune_still_refuses_test_split_via_cli(scratch_env: Path) -> None:
    result = runner.invoke(app, ["tune", "--split", "test", "--run-id", "whatever"])
    assert result.exit_code == 1


def test_ablate_command_orchestrates_arms_and_repeats(scratch_env: Path, fake_provider, monkeypatch) -> None:
    """Mocks the pipeline internals (already covered by test_extract.py /
    test_resolve.py) to isolate `ablate`'s own responsibility: looping
    arms x repeats, naming runs, and constructing the one legitimate
    `TestUnlock` outside `eval`."""
    text = "placeholder"
    _seed_document_and_chunk(scratch_env, text)

    calls = []

    def fake_extract_corpus(store, config, run_id, doc_ids=None):
        calls.append(("extract", config.arm, run_id))
        from c2k.extract import ExtractionStats

        return ExtractionStats()

    def fake_resolve_run(store, config, run_id):
        calls.append(("resolve", config.arm, run_id))
        from c2k.resolve import ResolveStats

        return ResolveStats()

    import c2k.cli as cli_mod

    monkeypatch.setattr(cli_mod.extract_mod, "extract_corpus", fake_extract_corpus)
    monkeypatch.setattr(cli_mod.resolve_pipeline, "resolve_run", fake_resolve_run)

    from c2k.eval.metrics import CostSummary, PRF1, RunMetrics

    def fake_evaluate_run(store, run_id, split, *, unlock=None, **kwargs):
        assert unlock is not None  # ablate must always unlock the test split
        f1 = 0.6 if run_id.startswith("a2") else 0.3
        return RunMetrics(
            run_id=run_id,
            triple_loose=PRF1(f1, f1, f1),
            triple_span_strict=PRF1(f1, f1, f1),
            hallucination=0.1,
            hallucination_llm=None,
            entailment_kappa=None,
            er_pairwise=None,
            schema_recall_ceiling=0.9,
            span_localization_failure_rate=0.0,
            adjudication_rate=None,
            cost=CostSummary(0, 0, 0, 0.0, 1),
            n_predicted=1,
            n_gold_in_schema=1,
            n_gold_total=1,
        )

    monkeypatch.setattr(cli_mod.eval_metrics, "evaluate_run", fake_evaluate_run)

    result = runner.invoke(app, ["ablate", "--arms", "a0,a2", "--repeats", "2"])
    assert result.exit_code == 0, result.output

    extract_calls = [c for c in calls if c[0] == "extract"]
    assert {c[2] for c in extract_calls} == {"a0_r0", "a0_r1", "a2_r0", "a2_r1"}
    resolve_calls = [c for c in calls if c[0] == "resolve"]
    # a0 has resolution.enabled == False (§8); only a2's runs call resolve.
    assert {c[2] for c in resolve_calls} == {"a2_r0", "a2_r1"}

    assert "success criterion" in result.output
    assert "PASS" in result.output  # a2 f1=0.6 clearly beats a0 f1=0.3, hallucination tied


def test_tune_command_sweeps_thresholds_without_touching_config(scratch_env: Path, fake_provider, monkeypatch) -> None:
    import numpy as np

    from c2k.resolve import blocking
    from c2k.resolve.adjudicate import PairVerdict

    config = load_config()
    store = Store(config.paths.db, canonical_dir=_canonical_dir(config))
    store.initialize()

    doc_id = "doc1"
    text = "Siemens AG acquired Mentor Grafiks for a large sum in 2017."
    store.insert_document(doc_id, f"{doc_id}.pdf", "sha", len(text), "tune")
    store.save_canonical_text(doc_id, text)

    store.start_run("run1", "a2", 0, "fake-model", "{}")
    store.insert_mention("m_subj", "run1", doc_id, "Siemens", 0, 7)
    store.insert_mention("m_obj", "run1", doc_id, "Mentor Grafiks", 21, 35)
    triple = Triple(
        triple_id="t1",
        run_id="run1",
        doc_id=doc_id,
        chunk_id="doc1:0",
        subject_mention_id="m_subj",
        object_mention_id="m_obj",
        relation=Relation.acquired,
        evidence_start=0,
        evidence_end=35,
        evidence_text=text[0:35],
        confidence=0.9,
    )
    store.insert_triple(triple)
    store.finish_run("run1")

    gold_path = Path(config.paths.gold) / f"{doc_id}.jsonl"
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    gold_path.write_text(
        '{"subject": "Siemens AG", "relation": "acquired", "object": "Mentor Graphics", '
        '"evidence_start": 0, "evidence_end": 35, "out_of_schema": false, "annotator": "t"}\n'
    )
    store.sync_gold_from_jsonl(config.paths.gold)
    store.close()

    monkeypatch.setattr(
        blocking, "embed_mentions", lambda mentions, canonical_by_doc, *, model_name: np.eye(len(mentions), dtype="float32")
    )
    # Every grid point that defers the (Siemens, Mentor Grafiks) pair asks the
    # same question; queue enough identical answers regardless of cache hits.
    for _ in range(20):
        fake_provider.append(BatchVerdicts(verdicts=[PairVerdict(same=True)]))

    config_before = (Path(scratch_env) / "config.yaml").read_text()
    result = runner.invoke(app, ["tune", "--split", "tune", "--run-id", "run1"])
    assert result.exit_code == 0, result.output
    assert "best:" in result.output
    assert (Path(scratch_env) / "config.yaml").read_text() == config_before  # tune never writes config


def test_tune_rejects_missing_run(scratch_env: Path) -> None:
    from c2k.schema import Relation  # noqa: F401 (ensures schema import path is warm)

    config = load_config()
    store = Store(config.paths.db, canonical_dir=_canonical_dir(config))
    store.initialize()
    store.insert_document("doc1", "doc1.pdf", "sha", 10, "tune")
    store.save_canonical_text("doc1", "x" * 10)
    store.close()

    result = runner.invoke(app, ["tune", "--split", "tune", "--run-id", "nonexistent"])
    assert result.exit_code == 1
