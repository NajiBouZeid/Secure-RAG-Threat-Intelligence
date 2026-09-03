"""Ports: the seams the benchmark swaps implementations across.

Phase 3 runs a matrix of (embedder x vector store x generator x defence set).
That is only a config sweep instead of copy-pasted scripts because every one of
those is a Protocol here and nothing above depends on a concrete class.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol, runtime_checkable

from threatrag.domain.models import (
    Answer,
    Chunk,
    Document,
    Principal,
    RetrievedChunk,
    SourceType,
)
from threatrag.domain.types import Matrix, Vector


@runtime_checkable
class Embedder(Protocol):
    """Text -> vector. Query and document encoding are separate calls because
    asymmetric models (GTR, E5, BGE) prefix them differently."""

    @property
    def name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> Matrix: ...

    def embed_query(self, text: str) -> Vector: ...


@runtime_checkable
class Chunker(Protocol):
    """Document -> chunks. Swappable so chunking strategy is a benchmark axis."""

    @property
    def name(self) -> str: ...

    def split(self, document: Document) -> list[Chunk]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Vector index with server-side payload filtering.

    ``search`` takes the principal so access control is enforced *inside* the
    query, not by discarding results afterwards.
    """

    def ensure_collection(self, vectors: Mapping[str, int]) -> None:
        """Declare every named vector the collection will ever hold.

        All of them up front, because a named vector cannot be added to an
        existing Qdrant collection after the fact -- only ``hnsw``/quantisation
        parameters are mutable. Points may still carry a subset of the vectors.
        """

    def upsert(self, vector_name: str, chunks: Sequence[Chunk], vectors: Matrix) -> int: ...

    def search(
        self,
        vector_name: str,
        query_vector: Vector,
        k: int,
        principal: Principal | None = None,
    ) -> list[RetrievedChunk]: ...

    def count(self) -> int: ...

    def delete_by_source_type(self, source_type: str) -> int: ...


@runtime_checkable
class Generator(Protocol):
    """LLM completion backend."""

    @property
    def model(self) -> str: ...

    def generate(self, system: str, user: str) -> str: ...


@runtime_checkable
class DocumentSource(Protocol):
    """A raw corpus (ATT&CK, NVD, vendor PDFs) yielding normalised documents."""

    @property
    def name(self) -> str: ...

    @property
    def source_type(self) -> SourceType:
        """Which corpus its documents belong to; what ``--reset`` deletes by."""
        ...

    def fetch(self) -> None:
        """Download raw material to disk. Idempotent."""

    def load(self) -> Iterable[Document]: ...


@runtime_checkable
class Defense(Protocol):
    """A single, independently toggleable mitigation.

    Each hook returns its input unchanged when the defence does not apply, so a
    defence set composes as a plain pipeline and every one can be measured in
    isolation.
    """

    @property
    def name(self) -> str: ...

    def on_ingest(self, document: Document) -> Document | None:
        """Return None to reject the document at the ingestion boundary."""
        return document

    def on_retrieve(self, retrieved: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return retrieved

    def on_answer(self, answer: Answer) -> Answer:
        return answer
