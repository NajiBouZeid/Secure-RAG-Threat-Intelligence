from __future__ import annotations

import pytest

from threatrag.domain.models import TLP, Document, SourceType, TrustTier
from threatrag.ingest.chunking import build_chunker

SECTIONED = """# T1055 - Process Injection

## Description
Adversaries may inject code into processes to evade defences.

## Detection
Monitor for API calls such as WriteProcessMemory and CreateRemoteThread.
"""


def make_document(text: str = SECTIONED, **overrides: object) -> Document:
    fields: dict[str, object] = {
        "id": "attack:T1055",
        "title": "T1055 Process Injection (technique)",
        "text": text,
        "source_type": SourceType.ATTACK_CTI,
        "source_ref": "T1055",
        "tlp": TLP.AMBER,
        "trust_tier": TrustTier.VENDOR,
    }
    fields.update(overrides)
    return Document(**fields)  # type: ignore[arg-type]


@pytest.mark.parametrize("strategy", ["structural", "recursive", "fixed"])
def test_every_strategy_produces_non_empty_chunks(strategy: str) -> None:
    chunks = build_chunker(strategy, 256, 32).split(make_document())
    assert chunks
    assert all(chunk.text.strip() for chunk in chunks)
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))


@pytest.mark.parametrize("strategy", ["structural", "recursive", "fixed"])
def test_security_fields_are_denormalised_onto_chunks(strategy: str) -> None:
    """Access control filters on chunk payloads, so a dropped field is a bypass."""
    for chunk in build_chunker(strategy, 256, 32).split(make_document()):
        assert chunk.tlp is TLP.AMBER
        assert chunk.trust_tier is TrustTier.VENDOR
        assert chunk.source_ref == "T1055"
        assert chunk.doc_id == "attack:T1055"


def test_structural_chunker_splits_on_sections_and_keeps_the_title() -> None:
    chunks = build_chunker("structural", 512, 64).split(make_document())
    assert len(chunks) == 2
    # Without the document header, a bare "Detection" block is unretrievable.
    assert all("T1055" in chunk.text for chunk in chunks)
    assert "Description" in chunks[0].text
    assert "Detection" in chunks[1].text


def test_structural_chunker_falls_back_when_there_are_no_sections() -> None:
    chunks = build_chunker("structural", 128, 16).split(make_document("plain text " * 60))
    assert len(chunks) > 1


@pytest.mark.parametrize("strategy", ["recursive", "fixed"])
def test_chunks_respect_the_size_budget(strategy: str) -> None:
    document = make_document("Adversaries persist across reboots. " * 200)
    for chunk in build_chunker(strategy, 200, 20).split(document):
        assert len(chunk.text) <= 200


def test_chunk_ids_are_content_addressed() -> None:
    """Edited text must not reuse a point id, or poisoned content survives re-ingest."""
    first = build_chunker("structural", 512, 64).split(make_document())
    second = build_chunker("structural", 512, 64).split(make_document())
    assert [c.id for c in first] == [c.id for c in second]

    edited = make_document(SECTIONED.replace("evade defences", "evade EDR"))
    changed = build_chunker("structural", 512, 64).split(edited)
    assert changed[0].id != first[0].id
