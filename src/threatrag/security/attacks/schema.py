"""The attack-file schema.

An attack is a versioned YAML document under ``attacks/``. It is deliberately
*data*, not code: it describes a poisoned document, the query that should
retrieve it, and the observable signal that means the attack landed. Nothing in
the file is executed -- it is ingested through the ordinary pipeline like any
other source, which is the whole point. The attack tests the real system, not a
mock of it.

The one invariant this schema enforces structurally is the M4 hard constraint:
an attack document is always ``SourceType.SYNTHETIC_ADVERSARIAL``. There is no
``source_type`` field to set, so an attack file *cannot* mislabel itself as NVD
or vendor content. Tampering with real records still attributed to their source
is out of scope by construction, not by convention.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from threatrag.domain.models import TLP, TrustTier


class AttackFamily(StrEnum):
    """The three attack classes M4 builds, each measured differently."""

    INJECTION = "injection"  # hijack the generator through retrieved context
    EXFILTRATION = "exfiltration"  # smuggle restricted content out via a rendered URL
    POISONING = "poisoning"  # win a top-k slot with attacker-chosen intelligence


class CriterionType(StrEnum):
    """What "it worked" is allowed to mean for a single check.

    Each attack declares one or more of these; the attack succeeds only when all
    of them hold, so a poisoning attack can require both that its document was
    retrieved *and* that its false claim reached the answer.
    """

    OUTPUT_CONTAINS = "output_contains"  # value is a substring of the answer text
    OUTPUT_MATCHES = "output_matches"  # value is a regex searched in the answer text
    RETRIEVED_TOPK = "retrieved_topk"  # the attack's own document was retrieved
    SINK_RECEIVED = "sink_received"  # the exfiltration sink logged value in a request


class SuccessCriterion(BaseModel):
    """One observable condition that contributes to an attack's verdict."""

    type: CriterionType
    # Meaning depends on ``type``; RETRIEVED_TOPK ignores it and checks the
    # attack's own document instead.
    value: str = ""
    case_sensitive: bool = False


class AttackDoc(BaseModel):
    """The poisoned document, before it is turned into a real indexed Document.

    ``tlp`` defaults to CLEAR so the poison is retrievable by any principal --
    an attacker cannot classify their own plant above the victim's clearance and
    still have it read. ``trust_tier`` defaults to UNTRUSTED because that is what
    the content honestly is; impersonating an authoritative source is done in the
    payload *text* (``Per MITRE...``), never by lying in this field.
    """

    title: str
    source_ref: str
    text: str
    tlp: TLP = TLP.CLEAR
    trust_tier: TrustTier = TrustTier.UNTRUSTED


class Attack(BaseModel):
    """A single red-team case."""

    id: str
    family: AttackFamily
    description: str
    target_query: str
    # The clearance the query runs under. Exfiltration needs a privileged
    # principal so a restricted note is in context to steal; injection and
    # poisoning land at the default CLEAR.
    clearance: TLP = TLP.CLEAR
    doc: AttackDoc
    success: list[SuccessCriterion] = Field(min_length=1)
