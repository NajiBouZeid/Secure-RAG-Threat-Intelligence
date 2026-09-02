"""Retrieval.

Deliberately hand-written rather than delegated to a framework. The retrieval
layer is the attack surface this project studies, so it has to be a layer we
own and can instrument -- a framework's ``RetrievalQA`` hides exactly the seam
where indirect prompt injection, retrieval poisoning and access-control
failures all live.
"""

from __future__ import annotations

from collections.abc import Sequence

from threatrag.domain.models import Principal, RetrievedChunk
from threatrag.domain.ports import Defense, Embedder, VectorStore


class Retriever:
    """Embed a question, search the index under the caller's authority, filter."""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        *,
        top_k: int = 5,
        score_threshold: float | None = None,
        defenses: Sequence[Defense] = (),
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._top_k = top_k
        self._score_threshold = score_threshold
        self._defenses = list(defenses)

    @property
    def top_k(self) -> int:
        return self._top_k

    def retrieve(
        self,
        question: str,
        *,
        k: int | None = None,
        principal: Principal | None = None,
    ) -> list[RetrievedChunk]:
        query_vector = self._embedder.embed_query(question)
        results = self._store.search(
            self._embedder.name, query_vector, k or self._top_k, principal=principal
        )

        if self._score_threshold is not None:
            results = [hit for hit in results if hit.score >= self._score_threshold]

        # Belt and braces: the store already filtered server-side, but an
        # adapter that ignores the principal must not silently become a bypass.
        if principal is not None:
            results = [hit for hit in results if principal.may_read(hit.chunk)]

        for defense in self._defenses:
            results = defense.on_retrieve(results)
        return results
