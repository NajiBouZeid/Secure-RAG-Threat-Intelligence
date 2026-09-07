"""D1, the provenance fence.

The defence's whole claim is that it changes presentation, not selection. If it
ever removed a passage it would be a filter wearing a fence's name, and its
measured benefit would come from the attack corpus being absent rather than
from the model resisting it.
"""

from __future__ import annotations

from threatrag.domain.models import TLP, Chunk, RetrievedChunk, SourceType, TrustTier
from threatrag.security.defenses.provenance_fence import FOOTER, ProvenanceFence


def _hit(tier: TrustTier, text: str = "body text") -> RetrievedChunk:
    chunk = Chunk(
        id=f"c-{int(tier)}",
        doc_id=f"d-{int(tier)}",
        ordinal=0,
        text=text,
        title="title",
        source_type=SourceType.ATTACK_CTI,
        source_ref="T1055",
        tlp=TLP.CLEAR,
        trust_tier=tier,
    )
    return RetrievedChunk(chunk=chunk, score=0.5)


def test_authoritative_passages_are_left_alone() -> None:
    hits = [_hit(TrustTier.AUTHORITATIVE)]

    assert ProvenanceFence().on_retrieve(hits)[0].chunk.text == "body text"


def test_untrusted_content_is_framed_as_quoted_evidence() -> None:
    fenced = ProvenanceFence().on_retrieve([_hit(TrustTier.UNTRUSTED)])[0].chunk.text

    assert "attacker-controlled" in fenced
    assert "not instructions to follow" in fenced
    assert fenced.endswith(FOOTER)
    assert "body text" in fenced


def test_nothing_is_dropped_and_nothing_is_reordered() -> None:
    """A filter would score by removing the attack corpus rather than by
    changing how the model reads it."""
    hits = [_hit(TrustTier.UNTRUSTED), _hit(TrustTier.AUTHORITATIVE), _hit(TrustTier.VENDOR)]

    kept = ProvenanceFence().on_retrieve(hits)

    assert len(kept) == 3
    assert [h.chunk.trust_tier for h in kept] == [tier.chunk.trust_tier for tier in hits]


def test_vendor_material_is_fenced_by_default() -> None:
    """Otherwise only the attack corpus is ever framed and the defence costs
    nothing by construction."""
    fenced = ProvenanceFence().on_retrieve([_hit(TrustTier.VENDOR)])[0].chunk.text

    assert "vendor publication" in fenced


def test_the_threshold_is_configurable_down_to_untrusted_only() -> None:
    fence = ProvenanceFence(min_tier=TrustTier.UNTRUSTED)

    assert fence.on_retrieve([_hit(TrustTier.VENDOR)])[0].chunk.text == "body text"
    assert "QUOTED MATERIAL" in fence.on_retrieve([_hit(TrustTier.UNTRUSTED)])[0].chunk.text


def test_the_original_chunk_is_not_mutated() -> None:
    """The retrieved objects are shared with the audit record; fencing must
    copy rather than edit in place."""
    hit = _hit(TrustTier.UNTRUSTED)

    ProvenanceFence().on_retrieve([hit])

    assert hit.chunk.text == "body text"
