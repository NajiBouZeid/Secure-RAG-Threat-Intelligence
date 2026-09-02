"""Ingestion pipeline: documents -> chunks -> vectors -> index.

Defences with an ``on_ingest`` hook run here, at the trust boundary. That
placement is the point: document sanitisation has to reject poisoned content
*before* it is embedded, because once a malicious chunk is in the index every
later control is reacting to something already retrievable.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field

from threatrag.domain.models import Chunk, Document
from threatrag.domain.ports import Chunker, Defense, Embedder, VectorStore


@dataclass(slots=True)
class IngestStats:
    documents_seen: int = 0
    documents_rejected: int = 0
    chunks_indexed: int = 0
    rejected_by: dict[str, int] = field(default_factory=dict)

    def merge(self, other: IngestStats) -> IngestStats:
        for name, count in other.rejected_by.items():
            self.rejected_by[name] = self.rejected_by.get(name, 0) + count
        return IngestStats(
            documents_seen=self.documents_seen + other.documents_seen,
            documents_rejected=self.documents_rejected + other.documents_rejected,
            chunks_indexed=self.chunks_indexed + other.chunks_indexed,
            rejected_by=self.rejected_by,
        )


def _batched(items: Sequence[Chunk], size: int) -> Iterator[Sequence[Chunk]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class IngestPipeline:
    """Wires a chunker, an embedder and a store into one idempotent ingest."""

    def __init__(
        self,
        chunker: Chunker,
        embedder: Embedder,
        store: VectorStore,
        *,
        vector_spec: Mapping[str, int] | None = None,
        defenses: Sequence[Defense] = (),
        batch_size: int = 256,
    ) -> None:
        self._chunker = chunker
        self._embedder = embedder
        self._store = store
        # Every named vector the collection will ever need, not just this run's:
        # Qdrant cannot add one to an existing collection later.
        self._vector_spec = dict(vector_spec or {embedder.name: embedder.dim})
        self._defenses = list(defenses)
        self._batch_size = batch_size

    def prepare(self) -> None:
        self._store.ensure_collection(self._vector_spec)

    def _admit(self, document: Document, stats: IngestStats) -> Document | None:
        """Run ingestion-time defences; ``None`` means the document was rejected."""
        current: Document | None = document
        for defense in self._defenses:
            if current is None:
                break
            current = defense.on_ingest(current)
            if current is None:
                stats.documents_rejected += 1
                stats.rejected_by[defense.name] = stats.rejected_by.get(defense.name, 0) + 1
                return None
        return current

    def run(self, documents: Iterable[Document]) -> IngestStats:
        self.prepare()
        stats = IngestStats()
        buffer: list[Chunk] = []

        for document in documents:
            stats.documents_seen += 1
            admitted = self._admit(document, stats)
            if admitted is None:
                continue
            buffer.extend(self._chunker.split(admitted))

            if len(buffer) >= self._batch_size:
                stats.chunks_indexed += self._flush(buffer)
                buffer = []

        stats.chunks_indexed += self._flush(buffer)
        return stats

    def _flush(self, chunks: Sequence[Chunk]) -> int:
        indexed = 0
        for batch in _batched(chunks, self._batch_size):
            vectors = self._embedder.embed_documents([chunk.text for chunk in batch])
            indexed += self._store.upsert(self._embedder.name, batch, vectors)
        return indexed
