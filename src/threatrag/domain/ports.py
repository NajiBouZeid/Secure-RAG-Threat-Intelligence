"""Ports: the seams the benchmark swaps implementations across.

Phase 3 runs a matrix of (embedder x vector store x generator x defence set).
That is only a config sweep instead of copy-pasted scripts because every one of
those is a Protocol here and nothing above depends on a concrete class.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
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

    @property
    def collection(self) -> str:
        """A human-readable name for what this store writes to, for messages
        that have to tell one index from another."""
        ...

    def ensure_collection(self, vectors: Mapping[str, int]) -> None:
        """Declare every named vector the collection will ever hold.

        All of them up front, because a named vector cannot be added to an
        existing Qdrant collection after the fact -- only ``hnsw``/quantisation
        parameters are mutable. Points may still carry a subset of the vectors.
        """

    def upsert(self, vector_name: str, chunks: Sequence[Chunk], vectors: Matrix) -> int:
        """Write chunks and their vectors, *replacing* any point with the same id.

        A point carrying only ``vector_name`` replaces one that held a different
        named vector, so this is not the way to add a second encoder's view of a
        corpus already in the index -- use ``attach_vectors`` for that.
        """

    def attach_vectors(self, vector_name: str, vectors: Mapping[str, Vector]) -> int:
        """Add one named vector to points that already exist, keyed by chunk id.

        Leaves every other named vector and the payload untouched, which
        ``upsert`` does not: this is what actually lets one chunk set carry both
        the MiniLM and the GTR embedding. Chunk ids with no stored point are
        skipped rather than created, since a vector without a payload is
        unretrievable and unattributable.
        """

    def scroll_chunks(
        self, *, source_type: str | None = None, batch_size: int = 256
    ) -> Iterator[Chunk]:
        """Enumerate stored chunks, optionally restricted to one corpus."""

    def get_vectors(self, vector_name: str, chunk_ids: Sequence[str]) -> dict[str, Vector]:
        """Read stored vectors back out by chunk id -- the attacker's view of a
        stolen index. Ids whose point lacks that named vector are omitted."""

    def search(
        self,
        vector_name: str,
        query_vector: Vector,
        k: int,
        principal: Principal | None = None,
    ) -> list[RetrievedChunk]: ...

    def count(self) -> int: ...

    def count_with_vector(self, vector_name: str) -> int:
        """How many points carry this named vector, rather than merely exist.

        A second encoder's vectors are attached to a subset of the corpus, and
        an ``upsert`` over those points drops them without erroring, so the
        count is the only evidence that a backfill is still intact.
        """

    def delete_by_source_type(self, source_type: str) -> int: ...


@runtime_checkable
class HybridSearch(Protocol):
    """A store that can fuse a dense ranking with a lexical one.

    Separate from ``VectorStore`` rather than a new argument to ``search``: a
    lexical ranking needs the question text, which no other store wants, and
    widening the port would put a parameter on every adapter that only one of
    them reads. ``hybrid`` is False on a store that has the method but no
    lexical index, so the retriever asks the store rather than its type.
    """

    @property
    def hybrid(self) -> bool: ...

    def search_hybrid(
        self,
        vector_name: str,
        query_vector: Vector,
        question: str,
        k: int,
        principal: Principal | None = None,
    ) -> list[RetrievedChunk]: ...


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
