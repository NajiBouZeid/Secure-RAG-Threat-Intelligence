"""Attaching a second encoder's vectors to chunks already in the index.

The inversion attack needs GTR vectors, but the system retrieves with MiniLM
and the M1-M4 numbers are measured against that index. So the GTR vectors are
added to the existing points rather than written as a fresh ingest: same chunk
ids, same payloads, same MiniLM vectors, one more named vector alongside.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from pydantic import BaseModel

from threatrag.domain.models import Chunk
from threatrag.domain.ports import Embedder, VectorStore
from threatrag.domain.types import Vector


class BackfillStats(BaseModel):
    vector_name: str
    chunks_seen: int = 0
    vectors_attached: int = 0

    @property
    def skipped(self) -> int:
        """Chunks whose point was not in the index -- a sample drawn from a
        collection that has since been re-ingested under a different chunker."""
        return self.chunks_seen - self.vectors_attached


def backfill_vectors(
    store: VectorStore,
    embedder: Embedder,
    chunks: Sequence[Chunk],
    *,
    batch_size: int = 32,
    on_batch: Callable[[int, int], None] | None = None,
) -> BackfillStats:
    """Embed each chunk with ``embedder`` and attach the result to its point.

    Batched because GTR is an order of magnitude heavier than MiniLM and a
    single call over the whole sample gives no progress signal on a CPU-only
    machine, where this is minutes rather than seconds.
    """
    stats = BackfillStats(vector_name=embedder.name)
    if not chunks:
        return stats

    for start in range(0, len(chunks), batch_size):
        batch = list(chunks[start : start + batch_size])
        matrix = embedder.embed_documents([chunk.text for chunk in batch])
        vectors: dict[str, Vector] = {
            chunk.id: vector for chunk, vector in zip(batch, matrix, strict=True)
        }
        attached = store.attach_vectors(embedder.name, vectors)

        stats = stats.model_copy(
            update={
                "chunks_seen": stats.chunks_seen + len(batch),
                "vectors_attached": stats.vectors_attached + attached,
            }
        )
        if on_batch is not None:
            on_batch(stats.chunks_seen, len(chunks))

    return stats
