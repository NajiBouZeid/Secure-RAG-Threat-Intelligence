"""D2, the per-document cap on top-k occupancy.

Two behaviours matter and they pull against each other: a poisoned document
must not be able to own the context window, and a legitimate document that
really is the best answer must not be silently truncated to nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from threatrag.domain.models import TLP, Chunk, Principal, RetrievedChunk, SourceType, TrustTier
from threatrag.rag.retriever import Retriever
from threatrag.security.defenses.source_cap import SourceCap


def _hit(doc_id: str, ordinal: int, score: float) -> RetrievedChunk:
    chunk = Chunk(
        id=f"{doc_id}#{ordinal}",
        doc_id=doc_id,
        ordinal=ordinal,
        text=f"{doc_id} passage {ordinal}",
        title=doc_id,
        source_type=SourceType.ATTACK_CTI,
        source_ref=doc_id,
        tlp=TLP.CLEAR,
        trust_tier=TrustTier.AUTHORITATIVE,
    )
    return RetrievedChunk(chunk=chunk, score=score)


def test_one_document_cannot_hold_every_slot() -> None:
    """poi-001 in miniature: the poison won all five, so the true corpus was
    not merely outranked but absent."""
    poisoned = [_hit("poison", i, 0.9 - i * 0.01) for i in range(5)]

    kept = SourceCap(max_per_document=2).on_retrieve(poisoned)

    assert len(kept) == 2


def test_the_passages_a_document_keeps_are_its_best_ones() -> None:
    hits = [_hit("d1", 0, 0.9), _hit("d1", 1, 0.8), _hit("d1", 2, 0.7)]

    kept = SourceCap(max_per_document=2).on_retrieve(hits)

    assert [hit.chunk.ordinal for hit in kept] == [0, 1]


def test_relevance_order_among_survivors_is_untouched() -> None:
    hits = [_hit("a", 0, 0.9), _hit("b", 0, 0.85), _hit("a", 1, 0.8), _hit("b", 1, 0.7)]

    kept = SourceCap(max_per_document=1).on_retrieve(hits)

    assert [(h.chunk.doc_id, h.score) for h in kept] == [("a", 0.9), ("b", 0.85)]


def test_a_cap_below_one_is_refused() -> None:
    """A cap of zero would empty every retrieval and read as a defence that
    stops all attacks."""
    with pytest.raises(ValueError, match="at least 1"):
        SourceCap(max_per_document=0)


class _Embedder:
    name = "all-minilm"

    def embed_query(self, text: str) -> np.ndarray:
        return np.ones(3, dtype=np.float32)


class _Store:
    """Returns a run of poison followed by the real corpus, as poi-001 did."""

    def __init__(self) -> None:
        self.asked_for = 0

    def search(
        self, vector_name: str, query_vector: object, k: int, principal: Principal | None = None
    ) -> list[RetrievedChunk]:
        self.asked_for = k
        ranked = [_hit("poison", i, 0.9 - i * 0.01) for i in range(5)]
        ranked += [_hit(f"real{i}", 0, 0.5 - i * 0.01) for i in range(10)]
        return ranked[:k]


def test_without_headroom_the_cap_only_deletes() -> None:
    """The honest failure mode, pinned: at overfetch=1 the retriever holds five
    results, the cap removes three, and the caller gets two passages instead of
    five. The poison is contained but the answer is starved."""
    retriever = Retriever(
        _Embedder(), _Store(), top_k=5, overfetch=1, defenses=[SourceCap(max_per_document=2)]
    )

    assert len(retriever.retrieve("q")) == 2


def test_with_headroom_the_cap_promotes_the_next_document() -> None:
    store = _Store()
    retriever = Retriever(
        _Embedder(), store, top_k=5, overfetch=3, defenses=[SourceCap(max_per_document=2)]
    )

    results = retriever.retrieve("q")

    assert store.asked_for == 15
    assert len(results) == 5
    assert [hit.chunk.doc_id for hit in results] == ["poison", "poison", "real0", "real1", "real2"]


def test_overfetch_defaults_to_reproducing_the_pre_m6_search() -> None:
    """Every M1-M5 number was measured at exactly top_k; the default must not
    quietly change the search that produced them."""
    store = _Store()

    Retriever(_Embedder(), store, top_k=5).retrieve("q")

    assert store.asked_for == 5
