"""Internal-notes corpus tests.

These assert properties of the shipped corpus itself, not just the loader. The
corpus is the only thing standing between "access control is implemented" and
"access control is demonstrated", so a note that silently loses its TLP would
quietly turn the end-to-end ACL demo back into a no-op.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from threatrag.domain.models import TLP, SourceType, TrustTier
from threatrag.ingest.sources.internal_notes import DEFAULT_PATH, InternalNotesSource

CORPUS = Path(DEFAULT_PATH)


def _documents() -> list:
    return list(InternalNotesSource(CORPUS).load())


def test_corpus_loads() -> None:
    documents = _documents()
    assert len(documents) >= 10
    assert all(doc.source_type is SourceType.INTERNAL_NOTE for doc in documents)


def test_ids_are_unique() -> None:
    documents = _documents()
    assert len({doc.id for doc in documents}) == len(documents)


def test_corpus_spans_several_clearance_levels() -> None:
    """A corpus that is uniformly one TLP cannot exercise the filter."""
    levels = {doc.tlp for doc in _documents()}
    assert TLP.RED in levels
    assert TLP.AMBER in levels
    assert len(levels) >= 3


def test_corpus_includes_low_trust_documents() -> None:
    """M4's injection work needs content that is readable but not authoritative."""
    tiers = {doc.trust_tier for doc in _documents()}
    assert TrustTier.COMMUNITY in tiers


def test_restricted_notes_are_not_readable_at_lower_clearance() -> None:
    from threatrag.domain.models import Chunk, Principal

    red = next(doc for doc in _documents() if doc.tlp is TLP.RED)
    chunk = Chunk(
        id="c1",
        doc_id=red.id,
        ordinal=0,
        text=red.text,
        title=red.title,
        source_type=red.source_type,
        source_ref=red.source_ref,
        tlp=red.tlp,
        trust_tier=red.trust_tier,
    )
    assert not Principal(id="a", clearance=TLP.AMBER).may_read(chunk)
    assert Principal(id="b", clearance=TLP.RED).may_read(chunk)


def test_missing_tlp_is_rejected_rather_than_defaulted(tmp_path: Path) -> None:
    """Defaulting a missing TLP to CLEAR would publish a restricted note."""
    path = tmp_path / "notes.yaml"
    path.write_text(
        yaml.safe_dump({"notes": [{"id": "X-1", "title": "t", "text": "b", "trust_tier": 1}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing"):
        list(InternalNotesSource(path).load())


def test_fictional_marker_present() -> None:
    """The file must stay obviously synthetic; it reads like real intelligence."""
    assert "FICTIONAL" in CORPUS.read_text(encoding="utf-8").upper()
