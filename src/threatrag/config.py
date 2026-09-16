"""Configuration loading.

A run is fully described by a YAML file plus environment overrides. Experiment
configs deep-merge over ``configs/base.yaml`` so a benchmark cell declares only
its deltas -- that is what keeps the Phase 3 sweep from becoming forty
near-identical files.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, Field, model_validator

from threatrag.domain.models import TLP, TrustTier

DEFAULT_CONFIG = Path("configs/base.yaml")
DEFAULT_ENV_FILE = Path(".env")


def load_dotenv(path: Path = DEFAULT_ENV_FILE) -> None:
    """Populate ``os.environ`` from a ``.env`` file, without overriding real env vars.

    Secrets -- currently just the NVD API key -- are read from the environment
    rather than from the YAML config, because the YAML is committed and the key
    must not be. A real environment variable always wins, so a container or CI
    runner can set the key without a file existing at all.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        # Deliberately setdefault: an explicit environment variable outranks the
        # file, and an empty placeholder in .env must not blank out a real value.
        value = value.strip().strip('"').strip("'")
        if key and value:
            os.environ.setdefault(key, value)


class EmbeddingModelConfig(BaseModel):
    backend: str = "sentence_transformers"
    model_id: str
    dim: int
    normalize: bool = True
    # Only meaningful for the mean_pooled_encoder backend: the token budget the
    # encoder sees. None means the model maximum.
    max_tokens: int | None = None


class EmbeddingConfig(BaseModel):
    primary: str
    models: dict[str, EmbeddingModelConfig]

    def resolve(self, name: str | None = None) -> tuple[str, EmbeddingModelConfig]:
        key = name or self.primary
        if key not in self.models:
            raise KeyError(f"Unknown embedding model {key!r}; have {sorted(self.models)}")
        return key, self.models[key]


class ChunkingConfig(BaseModel):
    strategy: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64


class VectorStoreConfig(BaseModel):
    backend: str = "qdrant"
    url: str = "http://localhost:6333"
    collection: str = "threatrag"
    # Used only when the corpus_segregation defence is enabled. Anything
    # classified above restrict_above is written here instead, so the public
    # collection an attacker steals holds no confidential vector at all.
    restricted_collection: str = "threatrag_restricted"
    restrict_above: TLP = TLP.GREEN


class Bm25Config(BaseModel):
    """The lexical half of hybrid retrieval. Written into the index at build time,
    so changing any of these means rebuilding the hybrid collection."""

    k1: float = 1.2
    b: float = 0.75
    # Mean tokens per indexed passage under the BM25 tokenizer. Fixed rather
    # than recomputed per write, so a document indexed later is weighted on the
    # same scale as the corpus it joins.
    avg_len: float = 100.0
    # Whether the chunk title is indexed with its text. Titles carry the
    # document's identifier ("T1127.003 JamPlus"); most chunk bodies after the
    # first do not.
    include_title: bool = False


class RetrievalConfig(BaseModel):
    top_k: int = 5
    score_threshold: float | None = None
    # Search depth multiplier. Retrieval fetches top_k * overfetch, defences run
    # over that, and the result is cut to top_k. 1 reproduces every pre-M6
    # number exactly; a retrieval-hook defence needs headroom to promote from.
    overfetch: int = 1
    # dense: MiniLM alone, every number before hybrid. hybrid: MiniLM and BM25
    # fused by reciprocal rank, which needs a collection built with sparse
    # vectors (`threatrag build-hybrid`).
    mode: Literal["dense", "hybrid"] = "dense"
    # Candidates each ranking contributes to the fusion. Not tuned on the gold set.
    hybrid_prefetch: int = 50
    bm25: Bm25Config = Field(default_factory=Bm25Config)


class GenerationConfig(BaseModel):
    backend: str = "ollama"
    url: str = "http://localhost:11434"
    model: str = "qwen2.5:7b"
    temperature: float = 0.0
    num_ctx: int = 8192
    # Character budget for retrieved passages. Ollama truncates an over-long
    # prompt silently, so this is bounded here where it can be recorded.
    max_context_chars: int = 12000
    # Output token budget. The mirror of num_ctx: without it a model can
    # generate until it fills the context window, which qwen2.5:1.5b does on at
    # least one benchmark cell. 1024 tokens is roughly 4000 characters, well
    # above any answer either model produces here.
    num_predict: int = 1024


class InjectionScreenConfig(BaseModel):
    """D4: reject documents that address the assistant rather than describe a threat."""

    # Extra regexes, applied case-insensitively on top of the built-in set.
    extra_patterns: list[str] = Field(default_factory=list)


class SourceCapConfig(BaseModel):
    """D2: how many passages one document may contribute to a top-k."""

    max_per_document: int = 2


class ProvenanceFenceConfig(BaseModel):
    """D1: the least-trusted tier that is still presented unframed.

    Defaults to VENDOR rather than UNTRUSTED on purpose. Fencing only the tier
    the attack corpus occupies would make the defence free by construction; the
    2729 vendor chunks are where its utility cost becomes visible.
    """

    min_tier: TrustTier = TrustTier.VENDOR


class EgressFilterConfig(BaseModel):
    """D3: hosts whose URLs may survive into a rendered answer.

    Empty by default. A permissive default would make the defence look free
    while leaving open hosts an attacker can reach, and the cost of severing
    citation links is exactly what M7 exists to price.
    """

    allowed_hosts: list[str] = Field(default_factory=list)


class CorroborationConfig(BaseModel):
    """D6: the trust tier at or below which a citation needs corroboration.

    COMMUNITY means authoritative and vendor sources stand on their own, and
    only community/untrusted material has to survive the check.
    """

    needs_support_at_or_below: TrustTier = TrustTier.COMMUNITY


class DefenseSettings(BaseModel):
    """Per-defence parameters.

    Separate from ``defenses`` on purpose: that list is pure membership, so a
    benchmark cell toggles a mitigation without restating its tuning, and a
    parameter sweep changes tuning without touching membership.
    """

    injection_screen: InjectionScreenConfig = Field(default_factory=InjectionScreenConfig)
    source_cap: SourceCapConfig = Field(default_factory=SourceCapConfig)
    provenance_fence: ProvenanceFenceConfig = Field(default_factory=ProvenanceFenceConfig)
    egress_filter: EgressFilterConfig = Field(default_factory=EgressFilterConfig)
    corroboration: CorroborationConfig = Field(default_factory=CorroborationConfig)


class PrincipalConfig(BaseModel):
    """A demo identity for the API and UI.

    Not authentication. The caller names the principal it wants and the server
    believes it, which is fine for a lab whose threat model is retrieval-layer
    access control rather than identity. A deployment would resolve these from
    a real identity provider; nothing else about the access-control path would
    change, because the Principal is already the only thing the store filters on.
    """

    label: str
    role: str = "analyst"
    clearance: TLP = TLP.CLEAR


class PathsConfig(BaseModel):
    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def eval_dir(self) -> Path:
        return self.data_dir / "eval"


class Config(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    embedding: EmbeddingConfig
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    principals: dict[str, PrincipalConfig] = Field(default_factory=dict)
    sources: dict[str, dict[str, Any]] = Field(default_factory=dict)
    defenses: list[str] = Field(default_factory=list)
    defense_settings: DefenseSettings = Field(default_factory=DefenseSettings)

    @model_validator(mode="after")
    def _hybrid_scores_are_ranks(self) -> Self:
        """Refuse the two configurations that would read an RRF score as a similarity.

        A fused score says where a passage ranked, not how close it is.
        ``corpus_segregation`` merges its two collections by score, which is
        exact for cosine similarity and meaningless for RRF: the merged top-k
        would quietly differ from a single-collection search, and M6's result
        that segregation leaves retrieval unchanged would stop being true
        without anything reporting it. A score threshold fails the same way.
        """
        if self.retrieval.mode != "hybrid":
            return self
        if "corpus_segregation" in self.defenses:
            raise ValueError(
                "retrieval.mode=hybrid cannot be combined with corpus_segregation: the "
                "segregated store merges collections by score, and a fused RRF score is a "
                "rank, not a similarity"
            )
        if self.retrieval.score_threshold is not None:
            raise ValueError(
                "retrieval.score_threshold has no meaning under retrieval.mode=hybrid: the "
                "fused score is a rank, not a similarity"
            )
        return self


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """Environment wins over YAML so the same config file works in and out of Docker."""
    env_map = {
        "THREATRAG_QDRANT_URL": ("vector_store", "url"),
        "THREATRAG_OLLAMA_URL": ("generation", "url"),
        "THREATRAG_DATA_DIR": ("paths", "data_dir"),
        "THREATRAG_LLM_MODEL": ("generation", "model"),
    }
    for env_key, (section, field) in env_map.items():
        value = os.getenv(env_key)
        if value:
            raw.setdefault(section, {})[field] = value
    return raw


def load_config(path: str | Path | None = None, overlay: str | Path | None = None) -> Config:
    """Load ``base.yaml`` (or ``path``), optionally deep-merging an experiment overlay."""
    load_dotenv()
    base_path = Path(path or os.getenv("THREATRAG_CONFIG") or DEFAULT_CONFIG)
    if not base_path.exists():
        raise FileNotFoundError(f"Config not found: {base_path}")

    raw = _read_yaml(base_path)
    if overlay is not None:
        raw = _deep_merge(raw, _read_yaml(Path(overlay)))
    return Config.model_validate(_apply_env_overrides(raw))
