"""Vector-store adapter tests.

The Qdrant client is faked, so what is under test is the adapter's contract
rather than the database. One behaviour here is load-bearing enough to be
pinned by name: adding a second encoder's vector must go through
``update_vectors``. Verified against a live Qdrant during M5, an ``upsert``
carrying a single named vector replaces the whole point and drops the other
vector -- which against the real collection would erase the MiniLM index that
every earlier milestone's numbers rest on.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from threatrag.domain.models import TLP, Chunk, SourceType, TrustTier
from threatrag.index import qdrant_store as qs
from threatrag.index.qdrant_store import QdrantVectorStore


def _chunk(ref: str, text: str = "body") -> Chunk:
    return Chunk(
        id=f"{ref}#0",
        doc_id=ref,
        ordinal=0,
        text=text,
        title=f"{ref} title",
        source_type=SourceType.ATTACK_CTI,
        source_ref=ref,
        tlp=TLP.CLEAR,
        trust_tier=TrustTier.AUTHORITATIVE,
    )


class FakeRecord:
    def __init__(self, point_id: str, vector: Any = None, payload: Any = None) -> None:
        self.id = point_id
        self.vector = vector
        self.payload = payload


class FakeClient:
    """Records calls; stores points as {point_id: {"vectors": {}, "payload": {}}}."""

    def __init__(self) -> None:
        self.points: dict[str, dict[str, Any]] = {}
        self.calls: list[str] = []
        self.scroll_pages: list[tuple[list[FakeRecord], Any]] = []

    def collection_exists(self, collection: str) -> bool:
        return True

    def upsert(self, collection: str, points: Any, wait: bool = True) -> None:
        self.calls.append("upsert")
        for point in points:
            # Mirrors Qdrant: the point is replaced wholesale.
            self.points[str(point.id)] = {"vectors": dict(point.vector), "payload": point.payload}

    def update_vectors(self, collection: str, points: Any, wait: bool = True) -> None:
        self.calls.append("update_vectors")
        for point in points:
            self.points[str(point.id)]["vectors"].update(point.vector)

    def retrieve(
        self,
        collection: str,
        ids: Any,
        with_payload: Any = True,
        with_vectors: Any = False,
    ) -> list[FakeRecord]:
        self.calls.append("retrieve")
        records = []
        for point_id in ids:
            stored = self.points.get(str(point_id))
            if stored is None:
                continue
            vector: Any = None
            if with_vectors:
                names = with_vectors if isinstance(with_vectors, list) else stored["vectors"]
                vector = {n: stored["vectors"][n] for n in names if n in stored["vectors"]}
            records.append(FakeRecord(str(point_id), vector, stored["payload"]))
        return records

    def count(self, collection: str, count_filter: Any = None, exact: bool = True) -> Any:
        self.calls.append("count")
        points = self.points.values()
        if count_filter is not None:
            wanted = count_filter.must[0].has_vector
            points = [p for p in points if wanted in p["vectors"]]
        return SimpleNamespace(count=len(list(points)))

    def scroll(
        self,
        collection: str,
        scroll_filter: Any = None,
        limit: int = 256,
        offset: Any = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> tuple[list[FakeRecord], Any]:
        self.calls.append("scroll")
        return self.scroll_pages.pop(0)


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> QdrantVectorStore:
    client = FakeClient()
    monkeypatch.setattr(qs, "QdrantClient", lambda **kwargs: client)
    return QdrantVectorStore("http://localhost:6333", "test")


def _client(store: QdrantVectorStore) -> FakeClient:
    return store._client  # type: ignore[return-value]


def test_attach_vectors_preserves_the_existing_named_vector(store: QdrantVectorStore) -> None:
    chunk = _chunk("T1055")
    store.upsert("all-minilm", [chunk], np.ones((1, 3), dtype=np.float32))

    written = store.attach_vectors("gtr-base", {chunk.id: np.full(4, 0.5, dtype=np.float32)})

    assert written == 1
    stored = next(iter(_client(store).points.values()))["vectors"]
    assert sorted(stored) == ["all-minilm", "gtr-base"]


def test_attach_vectors_never_upserts(store: QdrantVectorStore) -> None:
    """An upsert here would replace the point and drop the MiniLM vector."""
    chunk = _chunk("T1055")
    store.upsert("all-minilm", [chunk], np.ones((1, 3), dtype=np.float32))
    _client(store).calls.clear()

    store.attach_vectors("gtr-base", {chunk.id: np.zeros(4, dtype=np.float32)})

    assert "upsert" not in _client(store).calls
    assert "update_vectors" in _client(store).calls


def test_attach_vectors_skips_chunks_with_no_stored_point(store: QdrantVectorStore) -> None:
    """A vector without a payload is unretrievable and unattributable."""
    chunk = _chunk("T1055")
    store.upsert("all-minilm", [chunk], np.ones((1, 3), dtype=np.float32))

    written = store.attach_vectors(
        "gtr-base",
        {
            chunk.id: np.zeros(4, dtype=np.float32),
            "absent#0": np.zeros(4, dtype=np.float32),
        },
    )

    assert written == 1


def test_attach_vectors_is_a_no_op_when_given_nothing(store: QdrantVectorStore) -> None:
    assert store.attach_vectors("gtr-base", {}) == 0
    assert _client(store).calls == []


def test_get_vectors_keys_by_chunk_id_not_point_id(store: QdrantVectorStore) -> None:
    chunk = _chunk("T1055")
    store.upsert("all-minilm", [chunk], np.array([[1.0, 2.0, 3.0]], dtype=np.float32))

    found = store.get_vectors("all-minilm", [chunk.id])

    assert list(found) == [chunk.id]
    assert found[chunk.id].dtype == np.float32
    assert found[chunk.id].tolist() == [1.0, 2.0, 3.0]


def test_get_vectors_omits_points_lacking_that_vector(store: QdrantVectorStore) -> None:
    """Most of the index carries MiniLM only; a GTR read must not invent zeros."""
    chunk = _chunk("T1055")
    store.upsert("all-minilm", [chunk], np.ones((1, 3), dtype=np.float32))

    assert store.get_vectors("gtr-base", [chunk.id]) == {}


def test_scroll_chunks_pages_until_the_offset_is_exhausted(store: QdrantVectorStore) -> None:
    payloads = [
        QdrantVectorStore._payload(_chunk("T1055")),
        QdrantVectorStore._payload(_chunk("T1003")),
    ]
    _client(store).scroll_pages = [
        ([FakeRecord("a", payload=payloads[0])], "next"),
        ([FakeRecord("b", payload=payloads[1])], None),
    ]

    refs = [chunk.source_ref for chunk in store.scroll_chunks(batch_size=1)]

    assert refs == ["T1055", "T1003"]


def test_count_with_vector_counts_carriers_not_points(store: QdrantVectorStore) -> None:
    """Most of the index is MiniLM-only; the GTR backfill is a subset of it."""
    chunks = [_chunk("T1055"), _chunk("T1003")]
    store.upsert("all-minilm", chunks, np.ones((2, 3), dtype=np.float32))
    store.attach_vectors("gtr-base", {chunks[0].id: np.zeros(4, dtype=np.float32)})

    assert store.count() == 2
    assert store.count_with_vector("all-minilm") == 2
    assert store.count_with_vector("gtr-base") == 1


def test_count_with_vector_sees_the_drop_an_upsert_causes(store: QdrantVectorStore) -> None:
    """The M5 footgun, pinned: re-ingesting a backfilled chunk erases its GTR
    vector, and no write reports it. This count is the only witness."""
    chunk = _chunk("T1055")
    store.upsert("all-minilm", [chunk], np.ones((1, 3), dtype=np.float32))
    store.attach_vectors("gtr-base", {chunk.id: np.zeros(4, dtype=np.float32)})
    assert store.count_with_vector("gtr-base") == 1

    store.upsert("all-minilm", [chunk], np.ones((1, 3), dtype=np.float32))

    assert store.count_with_vector("gtr-base") == 0
    assert store.count() == 1
