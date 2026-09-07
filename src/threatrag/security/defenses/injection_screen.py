"""D4 -- refuse documents that address the assistant instead of describing a threat.

The earliest boundary a defence can act on: a document rejected here never
becomes a chunk, never earns a vector and can never be retrieved, so it defeats
an injection without the generator being involved at all.

**This screens text, never provenance.** Every document in ``attacks/`` carries
``trust_tier: UNTRUSTED``, correctly -- the tier is assigned by the system and
an attacker cannot forge it. A defence keyed on that label would reject the
whole attack corpus and nothing else, scoring 100% at zero cost, and would be
measuring our own labelling rather than any property of the attack. So the
signal here is the payload's own language and nothing else, which is what makes
the false-positive rate a real number: the patterns are free to fire on the
genuine corpus, and the M6 report says how often they do.

Two patterns were dropped after measuring them against the live 40815-chunk
corpus rather than reasoning about them: ``new task`` fires on T1053.005
Scheduled Task and on a phishing lure subject, and a bare ``append it`` fires on
DEADEYE.APPEND writing a payload to the end of a file. A corpus about
persistence discusses scheduling new tasks all day.

The remaining false-positive class is honest and unavoidable: threat
intelligence *about* prompt injection quotes injection payloads. CVE-2026-27001
describes an agent embedding an unsanitised working directory into its system
prompt, and this screen rejects it. A defence that blinds the system to
prompt-injection CVEs is a real cost, and it is the kind M7 exists to plot.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from threatrag.config import Config
from threatrag.domain.models import Document
from threatrag.security.defenses.base import BaseDefense

BUILTIN_PATTERNS: tuple[str, ...] = (
    # Overriding the instructions the model already holds.
    r"\bignore\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier)\b",
    r"\bdisregard\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier|the)\b",
    r"\b(?:system|initial|original)\s+prompt\b",
    r"\byour\s+(?:instructions|rules|guidelines|system\s+message)\b",
    r"\bas\s+an?\s+(?:ai|language\s+model|assistant)\b",
    r"\bnew\s+(?:instructions|directive)\b",
    # Meta-discourse about the reply itself. A threat report describes an
    # actor's behaviour; it has no reason to discuss the reader's answer.
    r"\byour\s+(?:answer|response|reply|output)\b",
    r"\b(?:every|each|the)\s+generated\s+response\b",
    r"\bmust\s+end\s+with\b",
    r"\bwhen\s+asked\b",
    r"\bthe\s+correct\s+answer\s+is\b",
    r"\bto\s+answer,?\s+first\b",
    r"\b(?:do\s+not|never)\s+(?:cite|mention|reveal|disclose)\b",
)


class InjectionScreen(BaseDefense):
    """Reject a document whose text instructs the reader rather than informing it."""

    name = "injection_screen"

    def __init__(self, extra_patterns: Sequence[str] = ()) -> None:
        self._patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in (*BUILTIN_PATTERNS, *extra_patterns)
        ]

    def matches(self, text: str) -> list[str]:
        """Every pattern that fires. Public for the report's false-positive census."""
        return [pattern.pattern for pattern in self._patterns if pattern.search(text)]

    def on_ingest(self, document: Document) -> Document | None:
        # The title is searched too. A chunker can split a document so a body
        # instruction lands in a chunk that is never retrieved, but the title is
        # denormalised onto every chunk the document produces.
        if self.matches(f"{document.title}\n{document.text}"):
            return None
        return document


def build(config: Config) -> InjectionScreen:
    return InjectionScreen(config.defense_settings.injection_screen.extra_patterns)
