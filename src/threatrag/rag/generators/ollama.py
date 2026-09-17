"""Ollama generation backend.

Ollama runs on the host rather than in compose (see docker-compose.yml), so the
only coupling is this HTTP client.

Three settings are deliberate rather than defaults. ``temperature`` is 0 because
the benchmark compares defence configurations, and a sampling temperature would
put run-to-run variance on top of the effect being measured. ``num_ctx`` is
pinned because Ollama silently truncates a prompt that exceeds the context
window: retrieved chunks would vanish from the middle of the prompt with no
error, and the answer would look like a retrieval failure instead of a
configuration one.

``num_predict`` bounds the *output*, which was missing and is the symmetric
control. Without it a model can generate until it exhausts the context window.
qwen2.5:1.5b does exactly that on one M7 cell -- poi-001's query under the full
defence set -- running past ten minutes and defeating three 180s retries, which
took down an otherwise complete sweep. That is not only a benchmark problem: an
assistant with no output bound will hang a real request the same way.
"""

from __future__ import annotations

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


class GenerationError(RuntimeError):
    """The backend could not produce a completion."""


class OllamaGenerator:
    """Wraps Ollama's ``/api/chat`` behind the ``Generator`` protocol."""

    def __init__(
        self,
        model: str,
        url: str = "http://localhost:11434",
        *,
        temperature: float = 0.0,
        num_ctx: int = 8192,
        num_predict: int = 1024,
        keep_alive: str = "60m",
        timeout: float = 180.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._model = model
        self._url = url.rstrip("/")
        self._temperature = temperature
        self._num_ctx = num_ctx
        self._num_predict = num_predict
        self._keep_alive = keep_alive
        self._timeout = timeout
        self._client = client

    @property
    def model(self) -> str:
        return self._model

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(base_url=self._url, timeout=self._timeout)
        return self._client

    @retry(
        # Retries cover a cold model load or a dropped connection, not a bad
        # request: an HTTPStatusError is deterministic and retrying it only
        # multiplies the wait before the same failure surfaces.
        retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def _post(self, payload: dict[str, object]) -> dict[str, object]:
        response = self._http().post("/api/chat", json=payload)
        response.raise_for_status()
        body: dict[str, object] = response.json()
        return body

    def generate(self, system: str, user: str) -> str:
        payload: dict[str, object] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "keep_alive": self._keep_alive,
            "options": {
                "temperature": self._temperature,
                "num_ctx": self._num_ctx,
                "num_predict": self._num_predict,
            },
        }

        try:
            body = self._post(payload)
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text.strip()
            if exc.response.status_code == httpx.codes.NOT_FOUND:
                raise GenerationError(
                    f"Ollama has no model {self._model!r}. Run `ollama pull {self._model}`."
                ) from exc
            raise GenerationError(f"Ollama returned {exc.response.status_code}: {detail}") from exc
        except httpx.HTTPError as exc:
            raise GenerationError(f"Cannot reach Ollama at {self._url}: {exc}") from exc

        message = body.get("message")
        if not isinstance(message, dict):
            raise GenerationError(f"Unexpected Ollama response shape: {body!r}")
        content = message.get("content")
        if not isinstance(content, str):
            raise GenerationError(f"Ollama returned no content: {message!r}")
        return content.strip()

    def load_state(self) -> dict[str, object] | None:
        """How this model is currently resident, or None if it is not loaded.

        ``size_vram`` below ``size`` means layers are on the CPU, which changes
        the arithmetic and therefore the answers: measured 2026-09-17, forcing a
        partial split rewrote 6 of 20 answers that were otherwise identical. The
        benchmark reads this to refuse a sweep whose cells would not be
        comparable, rather than discovering it afterwards in the variance.
        """
        try:
            response = self._http().get("/api/ps")
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError):
            return None
        models = body.get("models") if isinstance(body, dict) else None
        if not isinstance(models, list):
            return None
        for entry in models:
            if isinstance(entry, dict) and entry.get("model") == self._model:
                size, vram = entry.get("size"), entry.get("size_vram")
                return {
                    "size": size,
                    "size_vram": vram,
                    "fully_on_gpu": bool(size) and size == vram,
                    "digest": entry.get("digest"),
                }
        return None

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
