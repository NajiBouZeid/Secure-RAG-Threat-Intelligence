"""Retrieval quality metrics.

These are the utility axis of the Phase 3 benchmark. Every defence costs
something in retrieval quality, and a defence that stops an attack by making
the system useless is not a defence -- so attack-success-rate is only
interpretable when plotted against these numbers.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalScores:
    """Metrics for a single query."""

    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    hits: int
    relevant: int

    @property
    def hit(self) -> bool:
        return self.hits > 0


def score_query(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> RetrievalScores:
    """Score one ranked result list against a gold set of identifiers."""
    gold = set(relevant)
    top = list(retrieved[:k])

    if not gold:
        return RetrievalScores(0.0, 0.0, 0.0, 0.0, 0, 0)

    # Deduplicate for set-based metrics: several chunks of one document count once.
    unique_hits = {ref for ref in top if ref in gold}
    recall = len(unique_hits) / len(gold)
    precision = len(unique_hits) / len(top) if top else 0.0

    reciprocal_rank = 0.0
    for rank, ref in enumerate(top, start=1):
        if ref in gold:
            reciprocal_rank = 1.0 / rank
            break

    dcg = sum(1.0 / math.log2(rank + 1) for rank, ref in enumerate(top, start=1) if ref in gold)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
    ndcg = dcg / ideal if ideal else 0.0

    return RetrievalScores(
        recall_at_k=recall,
        precision_at_k=precision,
        reciprocal_rank=reciprocal_rank,
        ndcg_at_k=ndcg,
        hits=len(unique_hits),
        relevant=len(gold),
    )


@dataclass(frozen=True, slots=True)
class AggregateScores:
    queries: int
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    hit_rate: float

    def as_row(self) -> dict[str, float | int]:
        return {
            "queries": self.queries,
            "recall@k": round(self.recall_at_k, 4),
            "precision@k": round(self.precision_at_k, 4),
            "mrr": round(self.mrr, 4),
            "ndcg@k": round(self.ndcg_at_k, 4),
            "hit_rate": round(self.hit_rate, 4),
        }


def aggregate(scores: Sequence[RetrievalScores]) -> AggregateScores:
    """Macro-average over queries: every question counts equally regardless of gold-set size."""
    if not scores:
        return AggregateScores(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    n = len(scores)
    return AggregateScores(
        queries=n,
        recall_at_k=sum(s.recall_at_k for s in scores) / n,
        precision_at_k=sum(s.precision_at_k for s in scores) / n,
        mrr=sum(s.reciprocal_rank for s in scores) / n,
        ndcg_at_k=sum(s.ndcg_at_k for s in scores) / n,
        hit_rate=sum(1 for s in scores if s.hit) / n,
    )
