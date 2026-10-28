"""The single choke point for every model call (invariant I4). No other
module may import `anthropic` or `ollama`; both are imported lazily here,
inside the provider closures, so importing `c2k.llm` never requires either
SDK to be installed or configured — only calling through a live provider does.

`call()` is deliberately the exact signature from SPEC.md §7. `configure()`
supplies the `Store` and `Config` it needs, so tests can swap in a fake
provider (via `_PROVIDERS`) and a temp-file store without touching real
credentials.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Callable, TypeVar

from pydantic import BaseModel

from c2k.config import Config
from c2k.store import Store

T = TypeVar("T", bound=BaseModel)

Provider = Callable[..., tuple[BaseModel, dict]]

_store: Store | None = None
_config: Config | None = None
_api_key: str | None = None  # set only for a run that brings its own key


def configure(store: Store, config: Config, api_key: str | None = None) -> None:
    """`api_key`, if given, is used for Anthropic calls instead of the
    ANTHROPIC_API_KEY environment variable. It is kept only in this module,
    never written to the store or the config, and cleared by `reset()`."""
    global _store, _config, _api_key
    _store = store
    _config = config
    _api_key = api_key


def reset() -> None:
    global _store, _config, _api_key
    _store = None
    _config = None
    _api_key = None


def _make_sdk_provider(prefix: str, **request_kwargs) -> Provider:
    """Builds a provider for `{prefix}/{model}` via Instructor's unified
    `from_provider` interface. The import is inside the closure: it only
    fires the first time this provider is actually invoked. `request_kwargs`
    are passed to every request (used to bound the local model's calls)."""

    def _provider(
        prompt: str,
        response_model: type[BaseModel],
        *,
        model: str,
        temperature: float,
        max_retries: int,
    ) -> tuple[BaseModel, dict]:
        import instructor

        key_kwargs = {"api_key": _api_key} if (_api_key and prefix == "anthropic") else {}
        client = instructor.from_provider(f"{prefix}/{model}", **key_kwargs)
        result, completion = client.chat.completions.create_with_completion(
            messages=[{"role": "user", "content": prompt}],
            response_model=response_model,
            temperature=temperature,
            max_retries=max_retries,
            **request_kwargs,
        )
        usage = getattr(completion, "usage", None)
        input_tokens = 0
        output_tokens = 0
        if usage is not None:
            input_tokens = getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0)) or 0
            output_tokens = getattr(usage, "output_tokens", getattr(usage, "completion_tokens", 0)) or 0
        return result, {"input_tokens": input_tokens, "output_tokens": output_tokens}

    return _provider


_PROVIDERS: dict[str, Provider] = {
    "anthropic": _make_sdk_provider("anthropic"),
    # A small local model can fall into endless repetition; with no bound one
    # request hung a run for over an hour. A valid extraction is well under
    # 2048 tokens, so anything longer would fail validation anyway.
    "ollama": _make_sdk_provider("ollama", max_tokens=2048, timeout=300.0),
}


def _cache_key(model: str, prompt: str, schema_json: str, temperature: float, repeat_idx: int) -> str:
    raw = "|".join([model, prompt, schema_json, str(temperature), str(repeat_idx)])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def call(
    prompt: str, response_model: type[T], *, run_id: str, purpose: str
) -> T:
    if _store is None or _config is None:
        raise RuntimeError("c2k.llm.configure(store, config) must be called before call()")

    repeat_idx = _store.get_run_repeat_idx(run_id)
    model_name = _config.model.name
    provider_name = _config.model.provider
    temperature = _config.model.temperature  # temperature = 0 always (§7); config enforces this.

    schema_json = json.dumps(response_model.model_json_schema(), sort_keys=True)
    key = _cache_key(model_name, prompt, schema_json, temperature, repeat_idx)

    cached = _store.cache_get(key)
    if cached is not None:
        return response_model.model_validate(cached)

    provider = _PROVIDERS.get(provider_name)
    if provider is None:
        raise ValueError(f"unknown model provider: {provider_name!r}")

    started = time.monotonic()
    try:
        result, usage = provider(
            prompt,
            response_model,
            model=model_name,
            temperature=temperature,
            max_retries=_config.extraction.max_retries,
        )
    except Exception as exc:  # noqa: BLE001 - persistent schema failure must raise, not swallow (§7)
        raise RuntimeError(
            f"llm.call() failed for purpose={purpose!r} run_id={run_id!r} after retries: {exc}"
        ) from exc
    wall_seconds = time.monotonic() - started

    _store.record_call(run_id, usage.get("input_tokens", 0), usage.get("output_tokens", 0), wall_seconds)
    _store.cache_put(key, model_name, result.model_dump(mode="json"))
    return result
