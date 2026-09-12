"""Paired comparison of two cells that answered the same questions.

M7 compared cells by their aggregate utility, and the undefended baseline's own
spread (0.14-0.18 on 7b at 50 questions) swallowed every defence effect. Most of
that spread is which questions happen to flip between runs, and two cells that
answer the *same* questions can have it cancelled: score each question in both,
take the difference, and ask whether the mean difference is distinguishable
from zero.

What the interval does and does not cover has to be stated, because getting it
wrong is flattering. The bootstrap resamples *questions*, so it prices the
choice of question set and nothing else. Run-to-run variation of the generator
is not inside it -- a cell compared with an exact re-run of itself can still
produce an interval that excludes zero. That is why the baseline is repeated and
its repeats are compared with each other by this same statistic: the resulting
self-comparisons are the noise floor a defence effect must clear, and an
interval that excludes zero is not by itself a finding.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

DEFAULT_RESAMPLES = 10_000
# Fixed so a report regenerates bit-identically from the same JSONL.
DEFAULT_SEED = 20260912


@dataclass(frozen=True)
class PairedDelta:
    """Mean per-question difference ``treated - baseline``, with its interval."""

    n: int
    mean_delta: float
    low: float
    high: float
    # Questions the treated cell scored lower / higher on. Reported because a
    # mean of zero can hide a defence that loses ten questions and gains ten.
    lost: int
    gained: int

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0

    def as_row(self) -> dict[str, float | int]:
        return {
            "n": self.n,
            "mean_delta": round(self.mean_delta, 4),
            "low": round(self.low, 4),
            "high": round(self.high, 4),
            "lost": self.lost,
            "gained": self.gained,
        }


def paired_delta(
    baseline: Sequence[float],
    treated: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> PairedDelta:
    """Percentile bootstrap of the mean paired difference over questions."""
    if len(baseline) != len(treated):
        raise ValueError(f"{len(baseline)} baseline scores but {len(treated)} treated")
    if not baseline:
        raise ValueError("no paired scores to compare")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    base = np.asarray(baseline, dtype=np.float64)
    treat = np.asarray(treated, dtype=np.float64)
    deltas = treat - base
    n = len(deltas)

    rng = np.random.default_rng(seed)
    means = deltas[rng.integers(0, n, size=(resamples, n))].mean(axis=1)
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(means, [tail, 100 - tail])

    return PairedDelta(
        n=n,
        mean_delta=float(deltas.mean()),
        low=float(low),
        high=float(high),
        lost=int((deltas < 0).sum()),
        gained=int((deltas > 0).sum()),
    )
