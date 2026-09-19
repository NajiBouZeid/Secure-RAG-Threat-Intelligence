"""Read the committed evidence dumps back into a summary the UI can render.

Every number here is derived from a file in ``reports/data`` rather than
written down, because a figure quoted in two places drifts in one of them.
Nothing in this module runs a model, opens a socket or touches Qdrant: the
findings page has to work when the services are down, which on the machine
this was built on is most of the time.

What it deliberately does *not* do is decide anything. Where repeats of one
cell disagree, both the low and the high are reported and the cell is marked
unstable, rather than averaged into a number that looks more settled than the
measurement was.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel

M7_SWEEP = "m7_sweep.jsonl"
HYBRID_VARIANTS = "hybrid_attacks_variants.jsonl"
UTILITY_UNWARMED = "utility_sweep.jsonl"
UTILITY_WARMED = "utility_warm.jsonl"


class AttackCell(BaseModel):
    """One defence set against one corpus, over however many repeats it ran."""

    defense_set: str
    model: str
    corpus: str
    total: int
    landed_low: int
    landed_high: int
    repeats: int
    # Attacks that landed in some repeats and not others. Named rather than
    # counted: which attack is unstable is the interesting part, and M7's is
    # always the same one.
    unstable: list[str]


class RetrieverCell(BaseModel):
    retrieval: str
    defenses: str
    corpus: str
    total: int
    landed_low: int
    landed_high: int
    repeats: int
    unstable: list[str]


class UtilityCell(BaseModel):
    label: str
    model: str
    repeat: int
    utility: float
    recall: float
    warm: bool


class Findings(BaseModel):
    attacks: list[AttackCell]
    retrievers: list[RetrieverCell]
    utility: list[UtilityCell]
    # Which evidence files were found. A page that silently drops a section
    # because a file is missing looks the same as one with nothing to report.
    sources: dict[str, bool]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _summarise(runs: list[set[str]]) -> tuple[int, int, list[str]]:
    """Collapse repeats of one cell into a range plus the attacks that moved."""
    counts = [len(run) for run in runs]
    every = set().union(*runs) if runs else set()
    unstable = sorted(a for a in every if 0 < sum(a in run for run in runs) < len(runs))
    return min(counts, default=0), max(counts, default=0), unstable


def attack_cells(rows: list[dict[str, Any]]) -> list[AttackCell]:
    grouped: dict[tuple[str, str, str], list[set[str]]] = defaultdict(list)
    totals: dict[tuple[str, str, str], int] = {}
    for row in rows:
        for route in row.get("routes") or []:
            key = (row["label"], row["model"], route["corpus"])
            grouped[key].append(set(route["survivors"]))
            totals[key] = route["total"]
    cells = []
    for key, runs in grouped.items():
        low, high, unstable = _summarise(runs)
        cells.append(
            AttackCell(
                defense_set=key[0],
                model=key[1],
                corpus=key[2],
                total=totals[key],
                landed_low=low,
                landed_high=high,
                repeats=len(runs),
                unstable=unstable,
            )
        )
    return cells


def retriever_cells(rows: list[dict[str, Any]]) -> list[RetrieverCell]:
    grouped: dict[tuple[str, str, str], list[set[str]]] = defaultdict(list)
    totals: dict[tuple[str, str, str], int] = {}
    for row in rows:
        key = (row["retrieval"], row["defenses"], row["corpus"])
        grouped[key].append(set(row["survivors"]))
        totals[key] = row["total"]
    cells = []
    for key, runs in grouped.items():
        low, high, unstable = _summarise(runs)
        cells.append(
            RetrieverCell(
                retrieval=key[0],
                defenses=key[1],
                corpus=key[2],
                total=totals[key],
                landed_low=low,
                landed_high=high,
                repeats=len(runs),
                unstable=unstable,
            )
        )
    return cells


def utility_cells(rows: list[dict[str, Any]], *, warm: bool) -> list[UtilityCell]:
    return [
        UtilityCell(
            label=row["label"],
            model=row["model"],
            repeat=row.get("repeat", 1),
            utility=row["answers"]["answer_utility"],
            recall=row["answers"]["answer_recall"],
            warm=bool(row.get("warm", warm)),
        )
        for row in rows
    ]


def load(data_dir: Path) -> Findings:
    """Summarise whatever evidence is present in ``data_dir``."""
    files = {
        M7_SWEEP: _read_jsonl(data_dir / M7_SWEEP),
        HYBRID_VARIANTS: _read_jsonl(data_dir / HYBRID_VARIANTS),
        UTILITY_UNWARMED: _read_jsonl(data_dir / UTILITY_UNWARMED),
        UTILITY_WARMED: _read_jsonl(data_dir / UTILITY_WARMED),
    }
    return Findings(
        attacks=attack_cells(files[M7_SWEEP]),
        retrievers=retriever_cells(files[HYBRID_VARIANTS]),
        utility=(
            utility_cells(files[UTILITY_UNWARMED], warm=False)
            + utility_cells(files[UTILITY_WARMED], warm=True)
        ),
        sources={name: bool(rows) for name, rows in files.items()},
    )
