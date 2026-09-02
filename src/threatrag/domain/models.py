"""Core domain types.

Everything the rest of the system passes around is defined here, and nothing in
this module imports an adapter. The security-relevant fields -- ``tlp``,
``trust_tier`` -- live on the document from the moment of ingestion rather than
being bolted on in Phase 3, because the exfiltration attack is meaningless
without a classification model to violate.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import IntEnum, StrEnum

from pydantic import BaseModel, Field


class TLP(StrEnum):
    """Traffic Light Protocol classification, ordered from least to most restricted."""

    CLEAR = "clear"
    GREEN = "green"
    AMBER = "amber"
    RED = "red"

    @property
    def rank(self) -> int:
        return _TLP_ORDER[self]

    def readable_by(self, clearance: TLP) -> bool:
        return self.rank <= clearance.rank


_TLP_ORDER: dict[TLP, int] = {TLP.CLEAR: 0, TLP.GREEN: 1, TLP.AMBER: 2, TLP.RED: 3}


class TrustTier(IntEnum):
    """How much authority a document's *content* carries.

    Lower is more trusted. This is distinct from TLP: TLP governs who may read a
    document, trust tier governs how much the generator should let it steer an
    answer. Indirect prompt injection is fundamentally a trust-tier failure.
    """

    AUTHORITATIVE = 0  # MITRE ATT&CK, NVD -- structured, signed upstream
    VENDOR = 1  # Mandiant / CrowdStrike / Unit42 reports
    COMMUNITY = 2  # blogs, pastes, unvetted feeds
    UNTRUSTED = 3  # anything attacker-controllable


class SourceType(StrEnum):
    ATTACK_CTI = "attack_cti"
    NVD_CVE = "nvd_cve"
    VENDOR_REPORT = "vendor_report"
    SYNTHETIC_ADVERSARIAL = "synthetic_adversarial"


class Document(BaseModel):
    """A single retrievable source document, before chunking."""

    id: str
    title: str
    text: str
    source_type: SourceType
    source_ref: str = Field(description="Stable external identifier, e.g. T1055 or CVE-2024-3094")
    url: str | None = None
    tlp: TLP = TLP.CLEAR
    trust_tier: TrustTier = TrustTier.AUTHORITATIVE
    metadata: dict[str, str | list[str]] = Field(default_factory=dict)
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Chunk(BaseModel):
    """A retrieval unit.

    Access-control and provenance fields are denormalised from the parent
    document so the vector store can filter on them server-side. Filtering after
    retrieval leaks through the ranking: a restricted document that displaces a
    permitted one still changes what the user sees.
    """

    id: str
    doc_id: str
    ordinal: int
    text: str
    title: str
    source_type: SourceType
    source_ref: str
    url: str | None = None
    tlp: TLP = TLP.CLEAR
    trust_tier: TrustTier = TrustTier.AUTHORITATIVE
    metadata: dict[str, str | list[str]] = Field(default_factory=dict)

    @property
    def citation(self) -> str:
        return f"{self.source_ref} — {self.title}"


class Principal(BaseModel):
    """Who is asking. Drives retrieval-layer access control."""

    id: str
    role: str = "analyst"
    clearance: TLP = TLP.CLEAR
    allowed_source_types: frozenset[SourceType] | None = None

    def may_read(self, chunk: Chunk) -> bool:
        if not chunk.tlp.readable_by(self.clearance):
            return False
        return self.allowed_source_types is None or chunk.source_type in self.allowed_source_types


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float

    def __str__(self) -> str:
        return f"[{self.score:.3f}] {self.chunk.citation}"


class Answer(BaseModel):
    """A generated answer plus everything needed to audit it."""

    question: str
    text: str
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    model: str = ""
    defenses_applied: list[str] = Field(default_factory=list)
    blocked: bool = False
    block_reason: str | None = None
