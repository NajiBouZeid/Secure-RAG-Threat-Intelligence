from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from threatrag.config import load_config

BASE = Path("configs/base.yaml")


def test_base_config_loads() -> None:
    config = load_config(BASE)
    assert config.embedding.primary in config.embedding.models
    assert config.retrieval.top_k > 0


def test_overlay_deep_merges_instead_of_replacing_sections(tmp_path: Path) -> None:
    """A benchmark cell states only its deltas; unrelated keys must survive."""
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(yaml.safe_dump({"chunking": {"strategy": "fixed"}}), encoding="utf-8")

    config = load_config(BASE, overlay)
    base = load_config(BASE)
    assert config.chunking.strategy == "fixed"
    assert config.chunking.chunk_size == base.chunking.chunk_size


def test_environment_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same config file has to work both in and out of Docker."""
    monkeypatch.setenv("THREATRAG_QDRANT_URL", "http://qdrant:6333")
    assert load_config(BASE).vector_store.url == "http://qdrant:6333"


def test_unknown_embedding_model_fails_loudly() -> None:
    with pytest.raises(KeyError):
        load_config(BASE).embedding.resolve("does-not-exist")


def test_missing_config_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(Path("configs/nope.yaml"))
