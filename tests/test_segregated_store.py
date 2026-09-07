"""D5, corpus segregation.

The claim this defence rests on is that splitting the index into two
collections costs nothing in retrieval quality, because scores from one
embedder under one metric are globally comparable. That is asserted directly:
the segregated ranking must equal the single-collection ranking.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from threatrag.domain.models import TLP, Chunk, Principal, RetrievedChunk, SourceType, TrustTier
from threatrag.index.segregated_store import SegregatedStore


def _chunk(ref: str, tlp: TLP) -> Chunk:
    return Chunk(
        id=f"{ref}#0",
        doc_id=ref,
        ordinal=0,
        text=f"{ref} body",
        title=ref,
        source_type=SourceType.ATTACK_CTI,
        source_ref=ref,
        tlp=tlp,
        trust_tier=TrustTier.AUTHORITATIVE,
    )


class FakeStore:
    """An in-memory store that ranks by a score attached to each chunk id."""

    def __init__(self, name: str, scores: dict[str, float] | None = None) -> None:
        self._name = name
        self.chunks: list[Chunk] = []
        self.vectors: list[Any] = []
        self.scores = scores or {}
        self.searched = 0

    @property
    def collection(self) -> str:
        return self._name

    def ensure_collection(self, vectors: Any) -> None: ...

    def upsert(self, vector_name: str, chunks: Any, vectors: Any) -> int:
        self.chunks.extend(chunks)
        self.vectors.extend(list(vectors))
        return len(chunks)

    def attach_vectors(self, vector_name: str, vectors: Any) -> int:
        held = {c.id for c in self.chunks}
        return len([cid for cid in vectors if cid in held])

    def scroll_chunks(self, *, source_type: Any = None, batch_size: int = 256) -> Any:
        yield from self.chunks

    def get_vectors(self, vector_name: str, chunk_ids: Any) -> dict[str, Any]:
        held = {c.id for c in self.chunks}
        return {cid: np.zeros(3) for cid in chunk_ids if cid in held}

    def search(
        self, vector_name: str, query_vector: Any, k: int, principal: Principal | None = None
    ) -> list[RetrievedChunk]:
        self.searched += 1
        ranked = sorted(self.chunks, key=lambda c: self.scores.get(c.id, 0.0), reverse=True)
        return [RetrievedChunk(chunk=c, score=self.scores.get(c.id, 0.0)) for c in ranked[:k]]

    def count(self) -> int:
        return len(self.chunks)

    def count_with_vector(self, vector_name: str) -> int:
        return len(self.chunks)

    def delete_by_source_type(self, source_type: str) -> int:
        return 0


PUBLIC = [_chunk("T1055", TLP.CLEAR), _chunk("T1003", TLP.GREEN)]
SECRET = [_chunk("INT-2026-002", TLP.RED), _chunk("INT-2026-001", TLP.AMBER)]
SCORES = {"T1055#0": 0.90, "INT-2026-002#0": 0.85, "T1003#0": 0.70, "INT-2026-001#0": 0.60}


def _store() -> tuple[SegregatedStore, FakeStore, FakeStore]:
    public, restricted = FakeStore("pub", SCORES), FakeStore("res", SCORES)
    return SegregatedStore(public, restricted), public, restricted


def test_classification_decides_the_collection_not_the_corpus() -> None:
    """An AMBER vendor report is segregated for the same reason a note is, so
    this is a policy rather than a special case for one source."""
    store, public, restricted = _store()
    chunks = [*PUBLIC, *SECRET]

    store.upsert("all-minilm", chunks, np.arange(len(chunks) * 3).reshape(len(chunks), 3))

    assert [c.source_ref for c in public.chunks] == ["T1055", "T1003"]
    assert [c.source_ref for c in restricted.chunks] == ["INT-2026-002", "INT-2026-001"]


def test_each_chunk_keeps_its_own_vector_across_the_split() -> None:
    """The rows are reordered by the split; pairing them by position again
    would give every note somebody else's embedding."""
    store, public, restricted = _store()
    chunks = [PUBLIC[0], SECRET[0], PUBLIC[1]]
    vectors = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])

    store.upsert("all-minilm", chunks, vectors)

    assert public.vectors[0].tolist() == [1.0, 1.0, 1.0]
    assert public.vectors[1].tolist() == [3.0, 3.0, 3.0]
    assert restricted.vectors[0].tolist() == [2.0, 2.0, 2.0]


def test_a_stolen_public_index_holds_no_confidential_vector() -> None:
    """The whole point against M5: both attacks work on stored vectors, so the
    only thing that helps is the vector not being there."""
    store, public, _ = _store()
    chunks = [*PUBLIC, *SECRET]

    store.upsert("all-minilm", chunks, np.zeros((len(chunks), 3)))

    assert all(c.tlp.rank <= TLP.GREEN.rank for c in public.chunks)
    assert public.get_vectors("all-minilm", ["INT-2026-002#0"]) == {}


def test_the_merged_ranking_equals_the_single_collection_ranking() -> None:
    """Scores from one embedder under one metric are globally comparable, and a
    chunk in the global top-k is in its own collection's top-k -- so this
    defence costs nothing in retrieval quality."""
    store, _, _ = _store()
    chunks = [*PUBLIC, *SECRET]
    store.upsert("all-minilm", chunks, np.zeros((len(chunks), 3)))
    cleared = Principal(id="ir", label="IR lead", clearance=TLP.RED)

    ranked = store.search("all-minilm", np.zeros(3), 3, principal=cleared)

    assert [hit.chunk.source_ref for hit in ranked] == ["T1055", "INT-2026-002", "T1003"]


def test_an_uncleared_query_never_touches_the_restricted_collection() -> None:
    """Not queried and then filtered -- not queried."""
    store, _, restricted = _store()
    store.upsert("all-minilm", [*PUBLIC, *SECRET], np.zeros((4, 3)))
    analyst = Principal(id="a", label="Analyst", clearance=TLP.GREEN)

    store.search("all-minilm", np.zeros(3), 3, principal=analyst)

    assert restricted.searched == 0


def test_a_cleared_query_reaches_both() -> None:
    store, public, restricted = _store()
    store.upsert("all-minilm", [*PUBLIC, *SECRET], np.zeros((4, 3)))
    cleared = Principal(id="ir", label="IR lead", clearance=TLP.RED)

    store.search("all-minilm", np.zeros(3), 3, principal=cleared)

    assert (public.searched, restricted.searched) == (1, 1)


def test_counts_span_both_collections() -> None:
    store, _, _ = _store()
    store.upsert("all-minilm", [*PUBLIC, *SECRET], np.zeros((4, 3)))

    assert store.count() == 4
