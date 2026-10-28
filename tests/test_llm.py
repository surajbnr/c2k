"""Request bounds on the SDK providers in c2k/llm.py."""

from __future__ import annotations

from c2k import llm




def test_ollama_provider_bounds_each_request(monkeypatch) -> None:
    """Only the local provider gets a token cap and timeout; Claude's does not."""
    import sys
    import types

    from pydantic import BaseModel

    class _Resp(BaseModel):
        x: int = 1

    captured: dict = {}

    class _Completions:
        def create_with_completion(self, **kwargs):
            captured.update(kwargs)
            return _Resp(), types.SimpleNamespace(usage=None)

    fake = types.SimpleNamespace(
        from_provider=lambda name, **kw: types.SimpleNamespace(chat=types.SimpleNamespace(completions=_Completions()))
    )
    monkeypatch.setitem(sys.modules, "instructor", fake)

    llm._PROVIDERS["ollama"]("p", _Resp, model="m", temperature=0.0, max_retries=1)
    assert captured["max_tokens"] == 2048 and captured["timeout"] == 300.0

    captured.clear()
    llm._PROVIDERS["anthropic"]("p", _Resp, model="m", temperature=0.0, max_retries=1)
    assert "max_tokens" not in captured and "timeout" not in captured


def test_api_key_is_passed_only_when_configured_and_cleared_by_reset(monkeypatch) -> None:
    """A run that brings its own key gets it passed to the Anthropic client;
    the key never reaches the other providers and does not outlive reset()."""
    import sys
    import types

    from pydantic import BaseModel

    class _Resp(BaseModel):
        x: int = 1

    seen: list[dict] = []

    class _Completions:
        def create_with_completion(self, **kwargs):
            return _Resp(), types.SimpleNamespace(usage=None)

    def _from_provider(name, **kwargs):
        seen.append({"name": name, **kwargs})
        return types.SimpleNamespace(chat=types.SimpleNamespace(completions=_Completions()))

    monkeypatch.setitem(sys.modules, "instructor", types.SimpleNamespace(from_provider=_from_provider))
    call = lambda provider: llm._PROVIDERS[provider]("p", _Resp, model="m", temperature=0.0, max_retries=1)  # noqa: E731

    call("anthropic")
    assert "api_key" not in seen[-1]

    llm._api_key = "sk-test-123"
    try:
        call("anthropic")
        assert seen[-1]["api_key"] == "sk-test-123"
        call("ollama")
        assert "api_key" not in seen[-1]
    finally:
        llm.reset()
    call("anthropic")
    assert "api_key" not in seen[-1]
