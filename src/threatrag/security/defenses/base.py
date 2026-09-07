"""No-op hook defaults, so a defence can implement exactly one boundary.

:class:`~threatrag.domain.ports.Defense` declares three hooks and every defence
here acts at one of them. The protocol carries default bodies for documentation,
but structural typing still requires an implementer to *have* all three, and a
defence that hand-rolled two pass-throughs to satisfy the checker would bury the
one line that matters. Inheriting the defaults keeps each defence readable as
what it is: a single mitigation at a single boundary.
"""

from __future__ import annotations

from threatrag.domain.models import Answer, Document, RetrievedChunk


class BaseDefense:
    """Every hook returns its input unchanged; subclasses override exactly one."""

    #: Recorded on every Answer as ``defenses_applied``, and the key a config
    #: and a benchmark cell name this mitigation by.
    name: str = ""

    def on_ingest(self, document: Document) -> Document | None:
        return document

    def on_retrieve(self, retrieved: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return retrieved

    def on_answer(self, answer: Answer) -> Answer:
        return answer
