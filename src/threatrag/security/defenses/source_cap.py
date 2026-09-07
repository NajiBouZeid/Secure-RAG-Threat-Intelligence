"""D2 -- stop one document owning the whole context window.

M4's poi-001 stuffed keywords until its single poisoned document held all five
top-k slots, and with the true corpus entirely displaced the false attribution
was the only claim the generator could ground an answer in. A cap on how many
passages one document may contribute makes that displacement structurally
impossible, whatever the payload says.

The same mechanism answers a retrieval defect that predates the attack work:
near-duplicate chunks of one technique crowd the top-k (T1053.005 took three of
five slots on the APT29 persistence probe), so the caller pays for five
passages and receives three documents' worth of information. Defence and
retrieval fix are the same rule, which means M7 can measure whether it helps
answers even where nothing is attacking.

Capping is not free and the cost is not hypothetical. When a document genuinely
is the best answer -- a long technique page whose relevant detail is spread over
several chunks -- this drops its later passages in favour of weaker material
from elsewhere. That is the trade-off M7 plots, and it is why the cap runs
against over-fetched results: with headroom it *promotes* the next document,
without headroom it merely deletes.
"""

from __future__ import annotations

from collections import Counter

from threatrag.config import Config
from threatrag.domain.models import RetrievedChunk
from threatrag.security.defenses.base import BaseDefense


class SourceCap(BaseDefense):
    """Keep at most ``max_per_document`` passages from any one source document."""

    name = "source_cap"

    def __init__(self, max_per_document: int = 2) -> None:
        if max_per_document < 1:
            raise ValueError("max_per_document must be at least 1")
        self._cap = max_per_document

    def on_retrieve(self, retrieved: list[RetrievedChunk]) -> list[RetrievedChunk]:
        # Rank order is preserved, so the passages a document does keep are its
        # best ones: this changes which documents are represented, never the
        # relevance ordering among survivors.
        seen: Counter[str] = Counter()
        kept = []
        for hit in retrieved:
            doc_id = hit.chunk.doc_id
            if seen[doc_id] >= self._cap:
                continue
            seen[doc_id] += 1
            kept.append(hit)
        return kept


def build(config: Config) -> SourceCap:
    return SourceCap(config.defense_settings.source_cap.max_per_document)
