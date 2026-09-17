"""The generator's input and output bounds.

``num_ctx`` has been pinned since M2 because Ollama silently truncates an
over-long prompt. ``num_predict`` is the symmetric control and was missing:
without it a model generates until it fills the context window. qwen2.5:1.5b
does exactly that on one M7 cell -- poi-001's query under the full defence set
-- running past ten minutes and defeating three 180s retries, which took down an
otherwise complete sweep.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from threatrag.rag.generators.ollama import GenerationError, OllamaGenerator


class Recorder:
    """Captures the request body instead of reaching a model."""

    def __init__(self, content: str = "answer") -> None:
        self.payload: dict[str, Any] = {}
        self._content = content

    def post(self, path: str, json: dict[str, Any]) -> httpx.Response:
        self.payload = json
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": self._content}},
            request=httpx.Request("POST", f"http://test{path}"),
        )


def _generator(recorder: Recorder, **kwargs: Any) -> OllamaGenerator:
    gen = OllamaGenerator("qwen2.5:1.5b", **kwargs)
    gen._client = recorder  # type: ignore[assignment]
    return gen


def test_the_output_is_bounded_by_default() -> None:
    """An unbounded generation is a liveness bug in the product, not only in
    the benchmark: it hangs a real request the same way."""
    recorder = Recorder()

    _generator(recorder).generate("system", "user")

    assert recorder.payload["options"]["num_predict"] == 1024


def test_both_bounds_are_sent_together() -> None:
    recorder = Recorder()

    _generator(recorder, num_ctx=4096, num_predict=256).generate("system", "user")

    options = recorder.payload["options"]
    assert (options["num_ctx"], options["num_predict"]) == (4096, 256)
    assert options["temperature"] == 0.0


def test_the_model_is_kept_resident_between_requests() -> None:
    """Answers are stable for the lifetime of one model load and drift across
    loads, so a sweep must not let the model expire between cells."""
    recorder = Recorder()

    _generator(recorder).generate("system", "user")

    assert recorder.payload["keep_alive"] == "60m"


class Residency:
    """Answers /api/ps with a fixed residency report."""

    def __init__(self, models: list[dict[str, Any]]) -> None:
        self._models = models

    def get(self, path: str) -> httpx.Response:
        return httpx.Response(
            200,
            json={"models": self._models},
            request=httpx.Request("GET", f"http://test{path}"),
        )


def _load_state(models: list[dict[str, Any]]) -> dict[str, Any] | None:
    gen = OllamaGenerator("qwen2.5:1.5b")
    gen._client = Residency(models)  # type: ignore[assignment]
    return gen.load_state()


def test_a_fully_offloaded_model_is_reported_as_such() -> None:
    state = _load_state([{"model": "qwen2.5:1.5b", "size": 100, "size_vram": 100}])

    assert state is not None
    assert state["fully_on_gpu"] is True


def test_cpu_layers_are_detected() -> None:
    """The condition that quietly changes the arithmetic: measured 2026-09-17,
    forcing a partial split rewrote 6 of 20 otherwise identical answers."""
    state = _load_state([{"model": "qwen2.5:1.5b", "size": 100, "size_vram": 60}])

    assert state is not None
    assert state["fully_on_gpu"] is False


def test_another_model_being_resident_is_not_this_one() -> None:
    assert _load_state([{"model": "qwen2.5:7b", "size": 100, "size_vram": 100}]) is None


def test_a_timeout_is_reported_as_a_generation_error() -> None:
    """It is what an unbounded runaway looks like from the caller's side, and
    it must name the backend rather than surface a bare httpx error."""

    class Timeout:
        def post(self, path: str, json: dict[str, Any]) -> httpx.Response:
            raise httpx.ReadTimeout("timed out")

    gen = OllamaGenerator("qwen2.5:1.5b")
    gen._client = Timeout()  # type: ignore[assignment]

    with pytest.raises(GenerationError, match="Cannot reach Ollama"):
        gen.generate("system", "user")
