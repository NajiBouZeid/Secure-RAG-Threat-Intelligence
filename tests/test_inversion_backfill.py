"""Backfilling a second encoder's vectors onto an indexed corpus."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from threatrag.domain.models import TLP, Chunk, SourceType
from threatrag.domain.types import Matrix, Vector
from threatrag.security.inversion.backfill import backfill_vectors


def _chunk(index: int) -> Chunk:
    return Chunk(
        id=f"c-{index}#0",
        doc_id=f"c-{index}",
        ordinal=0,
        text=f"body {index}",
        title=f"title {index}",
        source_type=SourceType.ATTACK_CTI,
        source_ref=f"REF-{index}",
        tlp=TLP.CLEAR,
    )


class RecordingStore:
    """Accepts attachments for chunks it holds; reports the rest as skipped."""

    def __init__(self, known: set[str] | None = None) -> None:
        self.known = known
        self.attached: dict[str, Vector] = {}
        self.batches: list[int] = []
        self.upserts = 0

    def upsert(self, vector_name: str, chunks: object, vectors: object) -> int:
        self.upserts += 1
        return 0

    def attach_vectors(self, vector_name: str, vectors: Mapping[str, Vector]) -> int:
        self.batches.append(len(vectors))
        accepted = {
            chunk_id: vector
            for chunk_id, vector in vectors.items()
            if self.known is None or chunk_id in self.known
        }
        self.attached.update(accepted)
        return len(accepted)


class CountingEmbedder:
    name = "gtr-base"
    dim = 4

    def __init__(self) -> None:
        self.calls = 0

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        self.calls += 1
        return np.ones((len(texts), 4), dtype=np.float32)

    def embed_query(self, text: str) -> Vector:
        return np.ones(4, dtype=np.float32)


def test_every_chunk_gets_its_vector_attached() -> None:
    chunks = [_chunk(i) for i in range(10)]
    store = RecordingStore()

    stats = backfill_vectors(store, CountingEmbedder(), chunks, batch_size=4)

    assert stats.vectors_attached == 10
    assert sorted(store.attached) == sorted(chunk.id for chunk in chunks)


def test_backfill_never_upserts() -> None:
    """An upsert would replace the point and drop the MiniLM vector."""
    store = RecordingStore()

    backfill_vectors(store, CountingEmbedder(), [_chunk(0)])

    assert store.upserts == 0


def test_work_is_batched_rather_than_embedded_in_one_call() -> None:
    """GTR on CPU needs a progress signal; one call over the sample gives none."""
    embedder = CountingEmbedder()

    backfill_vectors(
        store := RecordingStore(), embedder, [_chunk(i) for i in range(10)], batch_size=4
    )

    assert store.batches == [4, 4, 2]
    assert embedder.calls == 3


def test_chunks_missing_from_the_index_are_counted_as_skipped() -> None:
    chunks = [_chunk(i) for i in range(4)]
    store = RecordingStore(known={"c-0#0", "c-1#0"})

    stats = backfill_vectors(store, CountingEmbedder(), chunks)

    assert stats.chunks_seen == 4
    assert stats.vectors_attached == 2
    assert stats.skipped == 2


def test_an_empty_sample_does_no_work() -> None:
    embedder = CountingEmbedder()

    stats = backfill_vectors(RecordingStore(), embedder, [])

    assert stats.vectors_attached == 0
    assert embedder.calls == 0


def test_progress_is_reported_per_batch() -> None:
    seen: list[tuple[int, int]] = []

    backfill_vectors(
        RecordingStore(),
        CountingEmbedder(),
        [_chunk(i) for i in range(5)],
        batch_size=2,
        on_batch=lambda done, total: seen.append((done, total)),
    )

    assert seen == [(2, 5), (4, 5), (5, 5)]
