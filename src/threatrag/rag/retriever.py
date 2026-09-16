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
from threatrag.domain.ports import Defense, Embedder, HybridSearch, VectorStore


class Retriever:
    """Embed a question, search the index under the caller's authority, filter."""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        *,
        top_k: int = 5,
        score_threshold: float | None = None,
        overfetch: int = 1,
        defenses: Sequence[Defense] = (),
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._top_k = top_k
        self._score_threshold = score_threshold
        self._overfetch = max(1, overfetch)
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
        wanted = k or self._top_k
        # A defence at this hook can only drop or reorder what it is handed, so
        # with no headroom a diversity rule is purely destructive: refusing a
        # duplicate leaves a hole rather than promoting the next document. The
        # default of 1 fetches exactly as before, so every pre-M6 number stands.
        query_vector = self._embedder.embed_query(question)
        depth = wanted * self._overfetch
        if isinstance(self._store, HybridSearch) and self._store.hybrid:
            results = self._store.search_hybrid(
                self._embedder.name, query_vector, question, depth, principal=principal
            )
        else:
            results = self._store.search(
                self._embedder.name, query_vector, depth, principal=principal
            )

        if self._score_threshold is not None:
            results = [hit for hit in results if hit.score >= self._score_threshold]

        # Belt and braces: the store already filtered server-side, but an
        # adapter that ignores the principal must not silently become a bypass.
        if principal is not None:
            results = [hit for hit in results if principal.may_read(hit.chunk)]

        for defense in self._defenses:
            results = defense.on_retrieve(results)
        # Truncated after the defences, never before: the over-fetched tail is
        # the material a diversity rule promotes from, and the caller must still
        # receive the k passages it asked for.
        return results[:wanted]
