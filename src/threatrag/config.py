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

DEFAULT_CONFIG = Path("configs/base.yaml")


class EmbeddingModelConfig(BaseModel):
    backend: str = "sentence_transformers"
    model_id: str
    dim: int
    normalize: bool = True


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


class GenerationConfig(BaseModel):
    backend: str = "ollama"
    url: str = "http://localhost:11434"
    model: str = "qwen2.5:7b"
    temperature: float = 0.0
    num_ctx: int = 8192
    # Character budget for retrieved passages. Ollama truncates an over-long
    # prompt silently, so this is bounded here where it can be recorded.
    max_context_chars: int = 12000


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
    sources: dict[str, dict[str, Any]] = Field(default_factory=dict)
    defenses: list[str] = Field(default_factory=list)


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
    base_path = Path(path or os.getenv("THREATRAG_CONFIG") or DEFAULT_CONFIG)
    if not base_path.exists():
        raise FileNotFoundError(f"Config not found: {base_path}")

    raw = _read_yaml(base_path)
    if overlay is not None:
        raw = _deep_merge(raw, _read_yaml(Path(overlay)))
    return Config.model_validate(_apply_env_overrides(raw))
