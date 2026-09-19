"""Findings reader tests.

The page these feed is the one most likely to be read by someone who will not
run the pipeline, so the property worth protecting is that it never reports a
cell as settled when its repeats disagreed.
"""

from __future__ import annotations

import json
from pathlib import Path

from threatrag.eval import findings


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def _sweep_row(label: str, model: str, survivors: list[str], repeat: int = 1) -> dict[str, object]:
    return {
        "label": label,
        "model": model,
        "repeat": repeat,
        "routes": [{"corpus": "m4", "landed": len(survivors), "total": 7, "survivors": survivors}],
        "answers": {"answer_utility": 0.18, "answer_recall": 0.06},
    }


def test_agreeing_repeats_report_one_number(tmp_path: Path) -> None:
    _write(
        tmp_path / findings.M7_SWEEP,
        [_sweep_row("none", "m", ["a"], r) for r in (1, 2, 3)],
    )
    cell = findings.load(tmp_path).attacks[0]
    assert (cell.landed_low, cell.landed_high) == (1, 1)
    assert cell.unstable == []
    assert cell.repeats == 3


def test_disagreeing_repeats_are_a_range_and_name_the_attack(tmp_path: Path) -> None:
    """An averaged 0.67 would look like a measurement; it is a coin flip."""
    _write(
        tmp_path / findings.M7_SWEEP,
        [
            _sweep_row("none", "m", ["flaky"], 1),
            _sweep_row("none", "m", [], 2),
            _sweep_row("none", "m", ["flaky"], 3),
        ],
    )
    cell = findings.load(tmp_path).attacks[0]
    assert (cell.landed_low, cell.landed_high) == (0, 1)
    assert cell.unstable == ["flaky"]


def test_missing_files_are_reported_not_hidden(tmp_path: Path) -> None:
    loaded = findings.load(tmp_path)
    assert loaded.attacks == []
    assert loaded.sources[findings.M7_SWEEP] is False


def test_warm_and_unwarmed_utility_rows_stay_distinguishable(tmp_path: Path) -> None:
    """They are different measurements and must never merge into one series."""
    _write(tmp_path / findings.UTILITY_UNWARMED, [_sweep_row("none", "m", [])])
    warm = _sweep_row("none", "m", [])
    warm["warm"] = True
    _write(tmp_path / findings.UTILITY_WARMED, [warm])
    cells = findings.load(tmp_path).utility
    assert sorted(c.warm for c in cells) == [False, True]


def test_retriever_cells_group_by_retriever_and_defence_set(tmp_path: Path) -> None:
    _write(
        tmp_path / findings.HYBRID_VARIANTS,
        [
            {
                "retrieval": "dense",
                "defenses": "none",
                "corpus": "m4",
                "repeat": r,
                "landed": 1,
                "total": 7,
                "survivors": ["x"],
            }
            for r in (1, 2)
        ]
        + [
            {
                "retrieval": "hybrid",
                "defenses": "none",
                "corpus": "m4",
                "repeat": 1,
                "landed": 2,
                "total": 7,
                "survivors": ["x", "y"],
            }
        ],
    )
    cells = {(c.retrieval, c.repeats): c for c in findings.load(tmp_path).retrievers}
    assert cells[("dense", 2)].landed_high == 1
    assert cells[("hybrid", 1)].landed_high == 2
