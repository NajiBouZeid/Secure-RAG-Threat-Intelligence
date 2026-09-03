"""Ollama generation backend.

Ollama runs on the host rather than in compose (see docker-compose.yml), so the
only coupling is this HTTP client.

Two settings are deliberate rather than defaults. ``temperature`` is 0 because
the benchmark compares defence configurations, and a sampling temperature would
put run-to-run variance on top of the effect being measured. ``num_ctx`` is
pinned because Ollama silently truncates a prompt that exceeds the context
window: retrieved chunks would vanish from the middle of the prompt with no
error, and the answer would look like a retrieval failure instead of a
configuration one.
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
        timeout: float = 180.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._model = model
        self._url = url.rstrip("/")
        self._temperature = temperature
        self._num_ctx = num_ctx
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
            "options": {"temperature": self._temperature, "num_ctx": self._num_ctx},
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

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
