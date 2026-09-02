"""Qdrant-backed vector store.

Two properties of Qdrant are load-bearing for this project rather than
incidental:

*Named vectors* let one collection hold both the MiniLM and the GTR embedding
of the same chunk. The Phase 2 embedding-inversion attack needs GTR (the only
strong encoder with a public vec2text corrector) while the system runs on
MiniLM, and comparing them is only meaningful over identical chunk sets.

*Server-side payload filtering* lets access control run inside the query. A
post-hoc filter would still be wrong: a restricted chunk that displaces a
permitted one changes the top-k the user receives even after it is dropped.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from threatrag.domain.models import TLP, Chunk, Principal, RetrievedChunk, SourceType, TrustTier
from threatrag.domain.types import Matrix, Vector

# Payload keys that carry security semantics; indexed so filtering stays cheap.
_TLP_KEY = "tlp"
_SOURCE_TYPE_KEY = "source_type"
_TRUST_TIER_KEY = "trust_tier"
_DOC_ID_KEY = "doc_id"

_KEYWORD_INDEXES = (_TLP_KEY, _SOURCE_TYPE_KEY, _DOC_ID_KEY)


def _point_id(chunk_id: str) -> str:
    """Qdrant point ids must be UUIDs or ints; derive a stable UUID from the chunk id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantVectorStore:
    """Vector store adapter implementing the ``VectorStore`` protocol."""

    def __init__(self, url: str, collection: str, *, timeout: int = 60) -> None:
        self._client = QdrantClient(url=url, timeout=timeout)
        self._collection = collection

    @property
    def collection(self) -> str:
        return self._collection

    def ensure_collection(self, vectors: Mapping[str, int]) -> None:
        """Create the collection with every named vector it will ever hold.

        Qdrant cannot add a named vector to an existing collection --
        ``update_collection`` only accepts ``VectorParamsDiff`` (hnsw and
        quantisation parameters). So all vector names are declared at creation
        time and points may carry any subset of them, which is what lets the
        MiniLM ingest and a later GTR ingest share one chunk set.
        """
        config = {
            name: qm.VectorParams(size=dim, distance=qm.Distance.COSINE)
            for name, dim in vectors.items()
        }

        if not self._client.collection_exists(self._collection):
            self._client.create_collection(self._collection, vectors_config=config)
        else:
            existing = self._client.get_collection(self._collection).config.params.vectors
            configured = set(existing) if isinstance(existing, dict) else set()
            missing = sorted(set(config) - configured)
            if missing:
                raise RuntimeError(
                    f"Collection {self._collection!r} exists without named vector(s) "
                    f"{missing}, and Qdrant cannot add them in place. Either use a "
                    f"different vector_store.collection or drop and re-ingest."
                )

        for field in _KEYWORD_INDEXES:
            self._client.create_payload_index(
                self._collection,
                field_name=field,
                field_schema=qm.PayloadSchemaType.KEYWORD,
                wait=True,
            )
        self._client.create_payload_index(
            self._collection,
            field_name=_TRUST_TIER_KEY,
            field_schema=qm.PayloadSchemaType.INTEGER,
            wait=True,
        )

    def upsert(self, vector_name: str, chunks: Sequence[Chunk], vectors: Matrix) -> int:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        if not chunks:
            return 0

        points = [
            qm.PointStruct(
                id=_point_id(chunk.id),
                vector={vector_name: vector.tolist()},
                payload=self._payload(chunk),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(self._collection, points=points, wait=True)
        return len(points)

    def search(
        self,
        vector_name: str,
        query_vector: Vector,
        k: int,
        principal: Principal | None = None,
    ) -> list[RetrievedChunk]:
        # Built explicitly rather than via .tolist(): numpy's stub returns a
        # nested-list union that the client signature does not accept.
        query: list[float] = [float(value) for value in query_vector]
        response = self._client.query_points(
            self._collection,
            query=query,
            using=vector_name,
            limit=k,
            query_filter=self._access_filter(principal),
            with_payload=True,
        )
        return [
            RetrievedChunk(chunk=self._to_chunk(point.payload or {}), score=float(point.score))
            for point in response.points
        ]

    def count(self) -> int:
        if not self._client.collection_exists(self._collection):
            return 0
        return int(self._client.count(self._collection, exact=True).count)

    def delete_by_source_type(self, source_type: str) -> int:
        """Remove a whole corpus. Used to reset poisoned documents between trials."""
        if not self._client.collection_exists(self._collection):
            return 0
        before = self.count()
        self._client.delete(
            self._collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key=_SOURCE_TYPE_KEY, match=qm.MatchValue(value=source_type)
                        )
                    ]
                )
            ),
            wait=True,
        )
        return before - self.count()

    def drop(self) -> None:
        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)

    @staticmethod
    def _access_filter(principal: Principal | None) -> qm.Filter | None:
        """Translate a principal into a Qdrant filter.

        Only classifications at or below the principal's clearance are made
        visible to the similarity search at all.
        """
        if principal is None:
            return None

        visible = [tlp.value for tlp in TLP if tlp.readable_by(principal.clearance)]
        conditions: list[qm.Condition] = [
            qm.FieldCondition(key=_TLP_KEY, match=qm.MatchAny(any=visible))
        ]
        if principal.allowed_source_types is not None:
            conditions.append(
                qm.FieldCondition(
                    key=_SOURCE_TYPE_KEY,
                    match=qm.MatchAny(
                        any=sorted(st.value for st in principal.allowed_source_types)
                    ),
                )
            )
        return qm.Filter(must=conditions)

    @staticmethod
    def _payload(chunk: Chunk) -> dict[str, Any]:
        return {
            "chunk_id": chunk.id,
            _DOC_ID_KEY: chunk.doc_id,
            "ordinal": chunk.ordinal,
            "text": chunk.text,
            "title": chunk.title,
            _SOURCE_TYPE_KEY: chunk.source_type.value,
            "source_ref": chunk.source_ref,
            "url": chunk.url,
            _TLP_KEY: chunk.tlp.value,
            _TRUST_TIER_KEY: int(chunk.trust_tier),
            "metadata": chunk.metadata,
        }

    @staticmethod
    def _to_chunk(payload: dict[str, Any]) -> Chunk:
        return Chunk(
            id=str(payload.get("chunk_id", "")),
            doc_id=str(payload.get(_DOC_ID_KEY, "")),
            ordinal=int(payload.get("ordinal", 0)),
            text=str(payload.get("text", "")),
            title=str(payload.get("title", "")),
            source_type=SourceType(payload.get(_SOURCE_TYPE_KEY, SourceType.ATTACK_CTI)),
            source_ref=str(payload.get("source_ref", "")),
            url=payload.get("url"),
            tlp=TLP(payload.get(_TLP_KEY, TLP.CLEAR)),
            trust_tier=TrustTier(int(payload.get(_TRUST_TIER_KEY, 0))),
            metadata=payload.get("metadata") or {},
        )
