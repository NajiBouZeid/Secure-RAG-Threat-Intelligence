"""Access-control unit tests.

These exist from M1 rather than M6 on purpose: the Phase 2 cross-context
exfiltration attack is only meaningful against a system that actually enforces
a classification model, so the model is built first and attacked later.
"""

from __future__ import annotations

import pytest

from threatrag.domain.models import TLP, Chunk, Principal, SourceType, TrustTier


def make_chunk(tlp: TLP, source_type: SourceType = SourceType.ATTACK_CTI) -> Chunk:
    return Chunk(
        id="c1",
        doc_id="d1",
        ordinal=0,
        text="body",
        title="title",
        source_type=source_type,
        source_ref="T1055",
        tlp=tlp,
        trust_tier=TrustTier.AUTHORITATIVE,
    )


@pytest.mark.parametrize(
    ("classification", "clearance", "expected"),
    [
        (TLP.CLEAR, TLP.CLEAR, True),
        (TLP.RED, TLP.CLEAR, False),
        (TLP.AMBER, TLP.RED, True),
        (TLP.RED, TLP.AMBER, False),
        (TLP.GREEN, TLP.GREEN, True),
    ],
)
def test_clearance_dominates_classification(
    classification: TLP, clearance: TLP, expected: bool
) -> None:
    principal = Principal(id="u", clearance=clearance)
    assert principal.may_read(make_chunk(classification)) is expected


def test_tlp_ordering_is_monotonic() -> None:
    order = [TLP.CLEAR, TLP.GREEN, TLP.AMBER, TLP.RED]
    assert [tlp.rank for tlp in order] == sorted(tlp.rank for tlp in order)


def test_source_type_allowlist_is_enforced_independently_of_clearance() -> None:
    """A cleared principal must still be confined to the corpora they are scoped to."""
    principal = Principal(
        id="contractor",
        clearance=TLP.RED,
        allowed_source_types=frozenset({SourceType.ATTACK_CTI}),
    )
    assert principal.may_read(make_chunk(TLP.RED, SourceType.ATTACK_CTI))
    assert not principal.may_read(make_chunk(TLP.CLEAR, SourceType.VENDOR_REPORT))


def test_no_allowlist_means_all_source_types() -> None:
    principal = Principal(id="analyst", clearance=TLP.AMBER)
    assert principal.may_read(make_chunk(TLP.AMBER, SourceType.VENDOR_REPORT))
