from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from threatrag.config import load_config, load_dotenv

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


def test_dotenv_populates_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text('# comment\nNVD_API_KEY="abc123"\n\nBLANK=\n', encoding="utf-8")
    monkeypatch.delenv("NVD_API_KEY", raising=False)

    load_dotenv(env_file)

    assert os.environ["NVD_API_KEY"] == "abc123"
    assert "BLANK" not in os.environ


def test_real_environment_outranks_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A container sets the key directly; a stale file must not shadow it."""
    env_file = tmp_path / ".env"
    env_file.write_text("NVD_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("NVD_API_KEY", "from-environment")

    load_dotenv(env_file)

    assert os.environ["NVD_API_KEY"] == "from-environment"


def test_dotenv_absent_is_not_an_error(tmp_path: Path) -> None:
    load_dotenv(tmp_path / "nope.env")


def test_overlays_compose_in_order(tmp_path: Path) -> None:
    """A cell built from a retrieval overlay and a defence overlay must carry both,
    with the later file winning where they overlap."""
    first = tmp_path / "first.yaml"
    first.write_text(
        yaml.safe_dump({"retrieval": {"top_k": 7, "overfetch": 2}, "defenses": ["source_cap"]}),
        encoding="utf-8",
    )
    second = tmp_path / "second.yaml"
    second.write_text(yaml.safe_dump({"retrieval": {"overfetch": 3}}), encoding="utf-8")

    config = load_config(BASE, [first, second])

    assert config.retrieval.top_k == 7
    assert config.retrieval.overfetch == 3
    assert config.defenses == ["source_cap"]
