"""Composition root.

Every concrete adapter is chosen here and nowhere else. Modules above depend
only on the protocols in :mod:`threatrag.domain.ports`, so swapping Qdrant for
Chroma or MiniLM for GTR is a config edit rather than a code change -- which is
what lets the Phase 3 benchmark sweep configurations instead of forking scripts.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import datetime

from threatrag.config import Config
from threatrag.domain.ports import Chunker, DocumentSource, Embedder, Generator, VectorStore
from threatrag.index.embedders.mean_pooled import MeanPooledEncoderEmbedder
from threatrag.index.embedders.sentence_transformer import SentenceTransformerEmbedder
from threatrag.index.qdrant_store import QdrantVectorStore
from threatrag.ingest.chunking import build_chunker
from threatrag.ingest.pipeline import IngestPipeline
from threatrag.ingest.sources.attack_cti import AttackCtiSource
from threatrag.ingest.sources.internal_notes import InternalNotesSource
from threatrag.ingest.sources.nvd_api import NvdClient
from threatrag.ingest.sources.nvd_cve import NvdCveSource
from threatrag.ingest.sources.vendor_report import VendorReportSource
from threatrag.rag.generators.ollama import OllamaGenerator
from threatrag.rag.pipeline import AnswerPipeline
from threatrag.rag.retriever import Retriever
from threatrag.security.attacks.runner import AttackRunner
from threatrag.security.attacks.sink import ExfiltrationSink


def build_embedder(
    config: Config, name: str | None = None, *, max_tokens: int | None = None
) -> Embedder:
    """Build an encoder. ``max_tokens`` overrides the config for one call, which
    is how the inversion control bundle is embedded at the corrector's own
    training length without declaring a second named vector."""
    key, spec = config.embedding.resolve(name)
    if spec.backend == "sentence_transformers":
        return SentenceTransformerEmbedder(
            name=key, model_id=spec.model_id, dim=spec.dim, normalize=spec.normalize
        )
    if spec.backend == "mean_pooled_encoder":
        return MeanPooledEncoderEmbedder(
            name=key,
            model_id=spec.model_id,
            dim=spec.dim,
            max_tokens=max_tokens if max_tokens is not None else spec.max_tokens,
        )
    raise ValueError(f"Unsupported embedding backend {spec.backend!r}")


def build_control_embedder(config: Config, name: str, max_tokens: int) -> MeanPooledEncoderEmbedder:
    """The encoder for the inversion control bundle, at a fixed token budget.

    Returns the concrete type rather than the ``Embedder`` protocol because the
    control bundle needs ``tokenized_prefix`` to record what the vector actually
    encoded, and it is only meaningful for the backend the corrector was trained
    against -- a sentence-transformers encoder here would silently produce a
    control in the wrong vector space, which is the failure this whole path
    exists to avoid.
    """
    embedder = build_embedder(config, name, max_tokens=max_tokens)
    if not isinstance(embedder, MeanPooledEncoderEmbedder):
        _, spec = config.embedding.resolve(name)
        raise ValueError(
            f"The inversion control bundle needs the mean_pooled_encoder backend, but "
            f"{name!r} is configured as {spec.backend!r}. The public corrector was trained "
            f"on mean-pooled encoder states; another convention inverts to nonsense."
        )
    return embedder


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


def build_nvd_source(config: Config) -> NvdCveSource:
    """The CVE corpus, with its ATT&CK cross-links resolved.

    The key is read from the environment and never from the config file, which
    is committed. Absent, the client throttles to the public rate limit and
    everything else behaves identically.
    """
    spec = config.sources.get("nvd_cve", {})
    client = NvdClient(config.paths.raw_dir / "nvd", api_key=os.getenv("NVD_API_KEY"))

    links: dict[str, list[str]] = {}
    if spec.get("include_attack_linked", True):
        # Needs the ATT&CK bundle on disk. Fetching CVEs without it would
        # silently drop the guaranteed-relevant half of the corpus, so let the
        # missing-file error surface instead.
        links = build_attack_source(config).cve_mentions()

    return NvdCveSource(
        client,
        attack_links=links,
        window_end=datetime.fromisoformat(str(spec.get("window_end", "2026-09-01"))),
        window_months=int(spec.get("window_months", 18)),
        severities=list(spec.get("severities", ["CRITICAL", "HIGH"])),
        recent_limit=int(spec.get("recent_limit", 5000)),
    )


def build_vendor_source(config: Config) -> VendorReportSource:
    spec = config.sources.get("vendor_report", {})
    manifest = spec.get("manifest")
    return (
        VendorReportSource(config.paths.raw_dir, manifest)
        if manifest
        else VendorReportSource(config.paths.raw_dir)
    )


def build_sources(config: Config) -> list[DocumentSource]:
    """Every corpus the config marks enabled, in ingestion order.

    Ordering is not cosmetic: the public corpus goes in first so that a partial
    ingest leaves the index public rather than leaving restricted notes sitting
    in a collection whose public half is missing.
    """
    builders: dict[str, Callable[[Config], DocumentSource]] = {
        "attack_cti": build_attack_source,
        "nvd_cve": build_nvd_source,
        "vendor_report": build_vendor_source,
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


def build_answer_pipeline(config: Config, embedder_name: str | None = None) -> AnswerPipeline:
    return AnswerPipeline(
        retriever=build_retriever(config, embedder_name),
        generator=build_generator(config),
        max_context_chars=config.generation.max_context_chars,
    )


def build_attack_runner(
    config: Config, *, sink: ExfiltrationSink | None = None, embedder_name: str | None = None
) -> AttackRunner:
    """The M4 harness: an answer pipeline plus an ingest pipeline over one store.

    Both pipelines share the store the real corpus lives in. The runner indexes a
    poison document, runs the query, and deletes the poison again, so attacks
    compete against the true corpus without leaving anything behind.
    """
    return AttackRunner(
        pipeline=build_answer_pipeline(config, embedder_name),
        ingest=build_pipeline(config, embedder_name),
        store=build_store(config),
        sink=sink,
    )


def build_retriever(config: Config, embedder_name: str | None = None) -> Retriever:
    return Retriever(
        embedder=build_embedder(config, embedder_name),
        store=build_store(config),
        top_k=config.retrieval.top_k,
        score_threshold=config.retrieval.score_threshold,
    )
