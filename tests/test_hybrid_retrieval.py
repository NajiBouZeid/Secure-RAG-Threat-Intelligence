"""Hybrid retrieval: store, retriever routing and the config refusals.

The Qdrant client is faked, so these pin what the adapter *asks* Qdrant for.
The failures they guard against all return plausible results: an access filter
applied to only one of the two rankings leaks through the other, a hybrid store
answering a dense search returns dense results under a hybrid label, and an RRF
score merged across collections as if it were a similarity reorders the top-k
without error.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml
from qdrant_client.http import models as qm

from threatrag.config import load_config
from threatrag.domain.models import TLP, Chunk, Principal, RetrievedChunk, SourceType, TrustTier
from threatrag.index import qdrant_store as qs
from threatrag.index.qdrant_store import SPARSE_VECTOR, QdrantVectorStore
from threatrag.index.sparse.bm25 import Bm25Encoder, term_id
from threatrag.rag.retriever import Retriever

BASE = Path("configs/base.yaml")


def _chunk(ref: str, text: str, title: str = "") -> Chunk:
    return Chunk(
        id=f"{ref}#1",
        doc_id=ref,
        ordinal=1,
        text=text,
        title=title or f"{ref} title",
        source_type=SourceType.ATTACK_CTI,
        source_ref=ref,
        tlp=TLP.CLEAR,
        trust_tier=TrustTier.AUTHORITATIVE,
    )


class FakeClient:
    def __init__(self, sparse_vectors: dict[str, Any] | None = None) -> None:
        self.upserts: list[Any] = []
        self.queries: list[dict[str, Any]] = []
        self.created: dict[str, Any] = {}
        self.exists = True
        self.sparse_vectors = sparse_vectors
        self.scroll_pages: list[tuple[list[Any], Any]] = []

    def collection_exists(self, collection: str) -> bool:
        return self.exists

    def create_collection(self, collection: str, **kwargs: Any) -> None:
        self.created = kwargs

    def create_payload_index(self, collection: str, **kwargs: Any) -> None:
        pass

    def get_collection(self, collection: str) -> Any:
        params = SimpleNamespace(
            vectors={"all-minilm": object()}, sparse_vectors=self.sparse_vectors
        )
        return SimpleNamespace(config=SimpleNamespace(params=params))

    def upsert(self, collection: str, points: Any, wait: bool = True) -> None:
        self.upserts.extend(points)

    def query_points(self, collection: str, **kwargs: Any) -> Any:
        self.queries.append(kwargs)
        return SimpleNamespace(points=[])

    def scroll(self, collection: str, **kwargs: Any) -> tuple[list[Any], Any]:
        return self.scroll_pages.pop(0)


def _store(
    monkeypatch: pytest.MonkeyPatch, client: FakeClient, *, hybrid: bool = True, title: bool = False
) -> QdrantVectorStore:
    monkeypatch.setattr(qs, "QdrantClient", lambda **kwargs: client)
    return QdrantVectorStore(
        "http://localhost:6333",
        "test",
        sparse=Bm25Encoder() if hybrid else None,
        sparse_title=title,
        prefetch=50,
    )


def test_hybrid_search_filters_both_rankings(monkeypatch: pytest.MonkeyPatch) -> None:
    """A filter on the dense prefetch alone would let the lexical ranking return
    a TLP:RED chunk to an analyst cleared for CLEAR."""
    client = FakeClient()
    store = _store(monkeypatch, client)
    analyst = Principal(id="a", clearance=TLP.CLEAR)

    store.search_hybrid("all-minilm", np.ones(3, dtype=np.float32), "T1055.001", 5, analyst)

    (query,) = client.queries
    dense, lexical = query["prefetch"]
    assert dense.using == "all-minilm"
    assert lexical.using == SPARSE_VECTOR
    assert dense.filter is not None
    assert dense.filter == lexical.filter
    assert isinstance(query["query"], qm.FusionQuery)
    assert query["query"].fusion == qm.Fusion.RRF
    assert lexical.query.indices == [term_id("t1055.001")]


def test_hybrid_prefetch_is_never_shallower_than_k(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    store = _store(monkeypatch, client)

    store.search_hybrid("all-minilm", np.ones(3, dtype=np.float32), "T1055", 80)

    assert [p.limit for p in client.queries[0]["prefetch"]] == [80, 80]
    assert client.queries[0]["limit"] == 80


def test_a_stopword_question_falls_back_to_the_dense_ranking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Qdrant rejects an empty sparse query; there is simply no lexical ranking."""
    client = FakeClient()
    store = _store(monkeypatch, client)

    store.search_hybrid("all-minilm", np.ones(3, dtype=np.float32), "what is the", 5)

    assert [p.using for p in client.queries[0]["prefetch"]] == ["all-minilm"]


def test_a_hybrid_store_refuses_a_dense_search(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store(monkeypatch, FakeClient())

    with pytest.raises(RuntimeError, match="search_hybrid"):
        store.search("all-minilm", np.ones(3, dtype=np.float32), 5)


def test_a_dense_store_searches_exactly_as_before(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dense collection is the control; hybrid support must not touch its query."""
    client = FakeClient()
    store = _store(monkeypatch, client, hybrid=False)

    store.search("all-minilm", np.ones(3, dtype=np.float32), 5)

    (query,) = client.queries
    assert "prefetch" not in query
    assert query["using"] == "all-minilm"


def test_upsert_writes_bm25_weights_only_when_hybrid(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = _chunk("T1055", "process injection")

    hybrid = FakeClient()
    _store(monkeypatch, hybrid).upsert("all-minilm", [chunk], np.ones((1, 3), dtype=np.float32))
    dense = FakeClient()
    _store(monkeypatch, dense, hybrid=False).upsert(
        "all-minilm", [chunk], np.ones((1, 3), dtype=np.float32)
    )

    assert sorted(hybrid.upserts[0].vector) == ["all-minilm", SPARSE_VECTOR]
    assert list(dense.upserts[0].vector) == ["all-minilm"]


def test_title_is_indexed_only_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Most chunk bodies do not carry their document's identifier; the title does."""
    chunk = _chunk("T1127.003", "Detection strategy", title="T1127.003 JamPlus (technique)")
    wanted = term_id("t1127.003")

    body_only = FakeClient()
    _store(monkeypatch, body_only).upsert("all-minilm", [chunk], np.ones((1, 3), np.float32))
    with_title = FakeClient()
    _store(monkeypatch, with_title, title=True).upsert(
        "all-minilm", [chunk], np.ones((1, 3), np.float32)
    )

    assert wanted not in body_only.upserts[0].vector[SPARSE_VECTOR].indices
    assert wanted in with_title.upserts[0].vector[SPARSE_VECTOR].indices


def test_new_hybrid_collection_declares_the_idf_modifier(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()
    client.exists = False
    store = _store(monkeypatch, client)

    store.ensure_collection({"all-minilm": 3})

    params = client.created["sparse_vectors_config"][SPARSE_VECTOR]
    assert params.modifier == qm.Modifier.IDF


def test_existing_dense_only_collection_is_refused_for_hybrid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fused against an empty lexical ranking, it would return dense results
    labelled hybrid."""
    store = _store(monkeypatch, FakeClient(sparse_vectors=None))

    with pytest.raises(RuntimeError, match="build-hybrid"):
        store.ensure_collection({"all-minilm": 3})


def test_copy_keeps_ids_payloads_and_dense_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Copied, not re-embedded: the dense half must be the control's own vectors."""
    source_client = FakeClient()
    payload = QdrantVectorStore._payload(_chunk("T1055", "process injection"))
    record = SimpleNamespace(
        id="p1", payload=payload, vector={"all-minilm": [0.1, 0.2], "gtr-base": [0.3]}
    )
    source_client.scroll_pages = [([record], None)]
    source = _store(monkeypatch, source_client, hybrid=False)
    target_client = FakeClient()
    target = _store(monkeypatch, target_client)

    assert target.copy_from(source) == 1

    (point,) = target_client.upserts
    assert point.id == "p1"
    assert point.payload == payload
    assert point.vector["all-minilm"] == [0.1, 0.2]
    assert point.vector["gtr-base"] == [0.3]
    assert term_id("injection") in point.vector[SPARSE_VECTOR].indices


class _Embedder:
    name = "all-minilm"
    dim = 3

    def embed_query(self, text: str) -> Any:
        return np.ones(3, dtype=np.float32)


class _HybridStore:
    def __init__(self, hybrid: bool) -> None:
        self.hybrid = hybrid
        self.called: list[str] = []

    def search(self, *args: Any, **kwargs: Any) -> list[RetrievedChunk]:
        self.called.append("search")
        return []

    def search_hybrid(self, *args: Any, **kwargs: Any) -> list[RetrievedChunk]:
        self.called.append(f"search_hybrid:{args[2]}")
        return []


@pytest.mark.parametrize(("hybrid", "expected"), [(True, "search_hybrid:T1055"), (False, "search")])
def test_retriever_asks_the_store_whether_it_is_hybrid(hybrid: bool, expected: str) -> None:
    store = _HybridStore(hybrid)

    Retriever(_Embedder(), store, top_k=5).retrieve("T1055")  # type: ignore[arg-type]

    assert store.called == [expected]


def _overlay(tmp_path: Path, content: dict[str, Any]) -> Path:
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(content), encoding="utf-8")
    return path


def test_hybrid_with_corpus_segregation_is_refused(tmp_path: Path) -> None:
    overlay = _overlay(
        tmp_path, {"retrieval": {"mode": "hybrid"}, "defenses": ["corpus_segregation"]}
    )

    with pytest.raises(ValueError, match="corpus_segregation"):
        load_config(BASE, overlay)


def test_hybrid_with_a_score_threshold_is_refused(tmp_path: Path) -> None:
    overlay = _overlay(tmp_path, {"retrieval": {"mode": "hybrid", "score_threshold": 0.5}})

    with pytest.raises(ValueError, match="score_threshold"):
        load_config(BASE, overlay)


def test_committed_hybrid_overlays_load() -> None:
    for name in ("retrieval_hybrid.yaml", "retrieval_hybrid_text.yaml"):
        config = load_config(BASE, Path("configs/experiments") / name)
        assert config.retrieval.mode == "hybrid"
        assert config.vector_store.collection != load_config(BASE).vector_store.collection
