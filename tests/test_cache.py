"""§7 / §15: same inputs + same repeat_idx hit the cache; a different
repeat_idx misses — this is what makes the 3-repeat variance design real
rather than three reads of one cached answer."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from c2k import llm
from c2k.config import (
    ChunkingConfig,
    Config,
    EvaluationConfig,
    ExtractionConfig,
    ModelConfig,
    PathsConfig,
    ResolutionConfig,
)
from c2k.store import Store


class _Reply(BaseModel):
    text: str


def _make_config(model_name: str = "fake-model") -> Config:
    return Config(
        arm="a0",
        model=ModelConfig(provider="fake", name=model_name, temperature=0.0),
        chunking=ChunkingConfig(),
        extraction=ExtractionConfig(),
        resolution=ResolutionConfig(),
        evaluation=EvaluationConfig(),
        paths=PathsConfig(),
    )


@pytest.fixture
def fake_provider(monkeypatch):
    calls = {"n": 0}

    def _provider(prompt, response_model, *, model, temperature, max_retries):
        calls["n"] += 1
        return response_model(text=f"reply-{calls['n']}"), {"input_tokens": 5, "output_tokens": 7}

    monkeypatch.setitem(llm._PROVIDERS, "fake", _provider)
    return calls


def test_same_repeat_idx_hits_cache(tmp_store: Store, fake_provider) -> None:
    config = _make_config()
    llm.configure(tmp_store, config)
    tmp_store.start_run("run-a", "a0", 0, config.model.name, config.model_dump_json())

    first = llm.call("prompt X", _Reply, run_id="run-a", purpose="test")
    second = llm.call("prompt X", _Reply, run_id="run-a", purpose="test")

    assert fake_provider["n"] == 1
    assert first == second


def test_different_repeat_idx_misses_cache(tmp_store: Store, fake_provider) -> None:
    config = _make_config()
    llm.configure(tmp_store, config)
    tmp_store.start_run("run-a", "a0", 0, config.model.name, config.model_dump_json())
    tmp_store.start_run("run-b", "a0", 1, config.model.name, config.model_dump_json())

    llm.call("prompt X", _Reply, run_id="run-a", purpose="test")
    llm.call("prompt X", _Reply, run_id="run-b", purpose="test")

    assert fake_provider["n"] == 2


def test_different_prompt_misses_cache(tmp_store: Store, fake_provider) -> None:
    config = _make_config()
    llm.configure(tmp_store, config)
    tmp_store.start_run("run-a", "a0", 0, config.model.name, config.model_dump_json())

    llm.call("prompt X", _Reply, run_id="run-a", purpose="test")
    llm.call("prompt Y", _Reply, run_id="run-a", purpose="test")

    assert fake_provider["n"] == 2


def test_call_records_usage_against_run(tmp_store: Store, fake_provider) -> None:
    config = _make_config()
    llm.configure(tmp_store, config)
    tmp_store.start_run("run-a", "a0", 0, config.model.name, config.model_dump_json())

    llm.call("prompt X", _Reply, run_id="run-a", purpose="test")

    run = tmp_store.get_run("run-a")
    assert run["n_calls"] == 1
    assert run["n_input_tokens"] == 5
    assert run["n_output_tokens"] == 7
