"""Generator adapter tests.

Driven through httpx's MockTransport rather than a live Ollama, so the error
paths -- the ones that decide whether a benchmark run fails loudly or silently
records a blank answer -- are actually exercised in CI.
"""

from __future__ import annotations

import re

import httpx
import pytest

from threatrag.rag.generators.ollama import GenerationError, OllamaGenerator


def _generator(handler: object, **kwargs: object) -> OllamaGenerator:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    client = httpx.Client(transport=transport, base_url="http://ollama.test")
    return OllamaGenerator(model="qwen2.5:7b", client=client, **kwargs)  # type: ignore[arg-type]


def test_returns_message_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "  hi  "}})

    assert _generator(handler).generate("sys", "user") == "hi"


def test_sends_system_and_user_turns_with_pinned_options() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "ok"}})

    _generator(handler, temperature=0.0, num_ctx=4096, num_predict=256).generate("SYS", "USER")

    assert seen["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]
    # Streaming off and pinned context bounds are what make a run reproducible.
    # num_predict is bounded for the same reason num_ctx is: the input was
    # capped and the output was not, and an unbounded generation can run until
    # it fills the window -- which took down an M7 sweep.
    assert seen["stream"] is False
    assert seen["options"] == {"temperature": 0.0, "num_ctx": 4096, "num_predict": 256}


def test_missing_model_names_the_pull_command() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model not found"})

    with pytest.raises(GenerationError, match=re.escape("ollama pull qwen2.5:7b")):
        _generator(handler).generate("sys", "user")


def test_unreachable_backend_raises_generation_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(GenerationError, match="Cannot reach Ollama"):
        _generator(handler).generate("sys", "user")


def test_malformed_response_is_an_error_not_an_empty_answer() -> None:
    """A blank answer would score as a wrong answer; it has to fail loudly."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"done": True})

    with pytest.raises(GenerationError):
        _generator(handler).generate("sys", "user")
