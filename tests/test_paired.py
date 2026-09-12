"""Paired per-question comparison between two benchmark cells."""

from __future__ import annotations

import pytest

from threatrag.eval.paired import paired_delta


def test_identical_cells_have_a_zero_delta_and_a_zero_interval() -> None:
    scores = [1.0, 0.0, 0.5, 1.0]

    delta = paired_delta(scores, scores)

    assert delta.mean_delta == 0.0
    assert (delta.low, delta.high) == (0.0, 0.0)
    assert delta.excludes_zero is False


def test_a_uniform_loss_is_detected() -> None:
    delta = paired_delta([1.0] * 40, [0.0] * 40)

    assert delta.mean_delta == -1.0
    assert delta.excludes_zero is True
    assert (delta.lost, delta.gained) == (40, 0)


def test_losses_and_gains_that_cancel_are_both_reported() -> None:
    """A mean of zero can hide a defence that loses ten questions and gains ten."""
    baseline = [1.0] * 10 + [0.0] * 10
    treated = [0.0] * 10 + [1.0] * 10

    delta = paired_delta(baseline, treated)

    assert delta.mean_delta == 0.0
    assert (delta.lost, delta.gained) == (10, 10)
    assert delta.excludes_zero is False


def test_the_interval_is_reproducible_from_the_same_scores() -> None:
    baseline = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    treated = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]

    assert paired_delta(baseline, treated) == paired_delta(baseline, treated)


def test_mismatched_lengths_are_refused() -> None:
    """Pairing only makes sense when both cells answered the same questions."""
    with pytest.raises(ValueError, match="3 baseline scores but 2 treated"):
        paired_delta([1.0, 0.0, 1.0], [1.0, 0.0])


def test_an_empty_comparison_is_refused() -> None:
    with pytest.raises(ValueError, match="no paired scores"):
        paired_delta([], [])
