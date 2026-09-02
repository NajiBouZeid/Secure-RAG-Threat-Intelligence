from __future__ import annotations

import math

from threatrag.eval.metrics import aggregate, score_query


def test_perfect_ranking_scores_one_across_the_board() -> None:
    scores = score_query(["T1", "T2", "T3"], ["T1", "T2", "T3"], k=3)
    assert scores.recall_at_k == 1.0
    assert scores.precision_at_k == 1.0
    assert scores.reciprocal_rank == 1.0
    assert math.isclose(scores.ndcg_at_k, 1.0)


def test_complete_miss_scores_zero() -> None:
    scores = score_query(["T9", "T8"], ["T1"], k=2)
    assert scores.recall_at_k == 0.0
    assert scores.reciprocal_rank == 0.0
    assert not scores.hit


def test_reciprocal_rank_tracks_the_first_hit() -> None:
    assert score_query(["A", "B", "T1"], ["T1"], k=3).reciprocal_rank == 1 / 3


def test_duplicate_hits_count_once_for_recall() -> None:
    """Several chunks of one technique must not inflate recall."""
    scores = score_query(["T1", "T1", "T1"], ["T1", "T2"], k=3)
    assert scores.recall_at_k == 0.5
    assert scores.hits == 1


def test_ndcg_rewards_ranking_relevant_results_higher() -> None:
    early = score_query(["T1", "X", "Y"], ["T1"], k=3).ndcg_at_k
    late = score_query(["X", "Y", "T1"], ["T1"], k=3).ndcg_at_k
    assert early > late


def test_results_below_k_are_truncated() -> None:
    assert score_query(["X", "Y", "T1"], ["T1"], k=2).recall_at_k == 0.0


def test_empty_gold_set_is_not_a_division_by_zero() -> None:
    assert score_query(["T1"], [], k=1).recall_at_k == 0.0


def test_aggregate_macro_averages_over_queries() -> None:
    summary = aggregate([score_query(["T1"], ["T1"], k=1), score_query(["X"], ["T1"], k=1)])
    assert summary.queries == 2
    assert summary.recall_at_k == 0.5
    assert summary.hit_rate == 0.5


def test_aggregate_of_nothing_is_zeroed_not_an_error() -> None:
    assert aggregate([]).queries == 0
