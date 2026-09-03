"""Composition root.

Every concrete adapter is chosen here and nowhere else. Modules above depend
only on the protocols in :mod:`threatrag.domain.ports`, so swapping Qdrant for
Chroma or MiniLM for GTR is a config edit rather than a code change -- which is
what lets the Phase 3 benchmark sweep configurations instead of forking scripts.
"""

from __future__ import annotations

from collections.abc import Callable

from threatrag.config import Config
from threatrag.domain.ports import Chunker, DocumentSource, Embedder, Generator, VectorStore
from threatrag.index.embedders.sentence_transformer import SentenceTransformerEmbedder
from threatrag.index.qdrant_store import QdrantVectorStore
from threatrag.ingest.chunking import build_chunker
from threatrag.ingest.pipeline import IngestPipeline
from threatrag.ingest.sources.attack_cti import AttackCtiSource
from threatrag.ingest.sources.internal_notes import InternalNotesSource
from threatrag.rag.generators.ollama import OllamaGenerator
from threatrag.rag.retriever import Retriever


def build_embedder(config: Config, name: str | None = None) -> Embedder:
    key, spec = config.embedding.resolve(name)
    if spec.backend != "sentence_transformers":
        raise ValueError(f"Unsupported embedding backend {spec.backend!r}")
    return SentenceTransformerEmbedder(
        name=key, model_id=spec.model_id, dim=spec.dim, normalize=spec.normalize
    )


def build_store(config: Config) -> VectorStore:
    backend = config.vector_store.backend
    if backend != "qdrant":
        raise ValueError(f"Unsupported vector store backend {backend!r}")
    return QdrantVectorStore(config.vector_store.url, config.vector_store.collection)


def build_generator(config: Config) -> Generator:
    backend = config.generation.backend
    if backend != "ollama":
        raise ValueError(f"Unsupported generation backend {backend!r}")
    return OllamaGenerator(
        model=config.generation.model,
        url=config.generation.url,
        temperature=config.generation.temperature,
        num_ctx=config.generation.num_ctx,
    )


def build_chunker_from_config(config: Config) -> Chunker:
    return build_chunker(
        config.chunking.strategy, config.chunking.chunk_size, config.chunking.chunk_overlap
    )


def build_attack_source(config: Config) -> AttackCtiSource:
    spec = config.sources.get("attack_cti", {})
    return AttackCtiSource(
        raw_dir=config.paths.raw_dir,
        url=str(spec["url"]),
        include_revoked=bool(spec.get("include_revoked", False)),
        include_deprecated=bool(spec.get("include_deprecated", False)),
    )


def build_internal_notes_source(config: Config) -> InternalNotesSource:
    spec = config.sources.get("internal_notes", {})
    path = spec.get("path")
    return InternalNotesSource(path) if path else InternalNotesSource()


def build_sources(config: Config) -> list[DocumentSource]:
    """Every corpus the config marks enabled, in ingestion order.

    Ordering is not cosmetic: the public corpus goes in first so that a partial
    ingest leaves the index public rather than leaving restricted notes sitting
    in a collection whose public half is missing.
    """
    builders: dict[str, Callable[[Config], DocumentSource]] = {
        "attack_cti": build_attack_source,
        "internal_notes": build_internal_notes_source,
    }
    sources: list[DocumentSource] = []
    for key, builder in builders.items():
        if config.sources.get(key, {}).get("enabled", False):
            sources.append(builder(config))
    return sources


def vector_spec(config: Config) -> dict[str, int]:
    """Every named vector the collection must declare at creation time."""
    return {name: spec.dim for name, spec in config.embedding.models.items()}


def build_pipeline(config: Config, embedder_name: str | None = None) -> IngestPipeline:
    return IngestPipeline(
        chunker=build_chunker_from_config(config),
        embedder=build_embedder(config, embedder_name),
        store=build_store(config),
        vector_spec=vector_spec(config),
    )


def build_retriever(config: Config, embedder_name: str | None = None) -> Retriever:
    return Retriever(
        embedder=build_embedder(config, embedder_name),
        store=build_store(config),
        top_k=config.retrieval.top_k,
        score_threshold=config.retrieval.score_threshold,
    )
