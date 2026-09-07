"""D5 -- the defence that is a topology, expressed as a toggle.

Unlike the other four this has no hook. It changes where chunks are *stored*,
not how a request is handled, and by the time any hook runs the decision has
already been made at ingest. The class exists so segregation is toggled from
the same ``defenses`` list as everything else -- a benchmark cell should not
have to reach into ``vector_store`` for one mitigation and into ``defenses``
for the rest -- and so it is recorded in ``defenses_applied`` on every answer it
shaped.

The work happens in :class:`~threatrag.index.segregated_store.SegregatedStore`,
which the composition root selects when this defence is enabled.
"""

from __future__ import annotations

from threatrag.config import Config
from threatrag.security.defenses.base import BaseDefense


class CorpusSegregation(BaseDefense):
    """Marker: confidential chunks live in their own collection."""

    name = "corpus_segregation"


def build(config: Config) -> CorpusSegregation:
    return CorpusSegregation()
