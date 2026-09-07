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
from typing import Any

import yaml
from pydantic import BaseModel, Field

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


class RetrievalConfig(BaseModel):
    top_k: int = 5
    score_threshold: float | None = None
    # Search depth multiplier. Retrieval fetches top_k * overfetch, defences run
    # over that, and the result is cut to top_k. 1 reproduces every pre-M6
    # number exactly; a retrieval-hook defence needs headroom to promote from.
    overfetch: int = 1


class GenerationConfig(BaseModel):
    backend: str = "ollama"
    url: str = "http://localhost:11434"
    model: str = "qwen2.5:7b"
    temperature: float = 0.0
    num_ctx: int = 8192
    # Character budget for retrieved passages. Ollama truncates an over-long
    # prompt silently, so this is bounded here where it can be recorded.
    max_context_chars: int = 12000


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
