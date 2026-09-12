"""Paired utility comparison over a utility-only benchmark JSONL.

Every cell is compared with the first undefended repeat of its own model,
question by question. The later undefended repeats go through the same
comparison, and those rows are the point of the table: they are an exact
configuration compared with itself, so whatever they show is noise, and a
defence row is only a finding when it clears them.

    python scripts/utility_report.py reports/data/utility_sweep.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from threatrag.eval.paired import PairedDelta, paired_delta

BASELINE = "none"


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    missing = [f"{row['label']}@{row['model']}" for row in rows if not row.get("records")]
    if missing:
        raise SystemExit(f"rows without per-question records cannot be paired: {missing}")
    return rows


def _by_question(row: dict[str, Any], field: str) -> dict[str, float]:
    return {record["id"]: float(record[field]) for record in row["records"]}


def compare(baseline: dict[str, Any], treated: dict[str, Any], field: str) -> PairedDelta:
    """Pair two cells on one per-question field, refusing mismatched question sets."""
    base = _by_question(baseline, field)
    treat = _by_question(treated, field)
    if base.keys() != treat.keys():
        raise SystemExit(
            f"{treated['label']}#{treated['repeat']} answered different questions from the "
            "baseline; pairing them would compare different things"
        )
    ids = sorted(base)
    return paired_delta([base[i] for i in ids], [treat[i] for i in ids])


def _interval(delta: PairedDelta) -> str:
    mark = " *" if delta.excludes_zero else ""
    return f"{delta.mean_delta:+.3f} [{delta.low:+.3f}, {delta.high:+.3f}]{mark}"


def report(rows: Sequence[dict[str, Any]]) -> str:
    lines: list[str] = []
    for model in sorted({row["model"] for row in rows}):
        cells = sorted(
            (row for row in rows if row["model"] == model),
            key=lambda row: (row["label"] != BASELINE, row["label"], row["repeat"]),
        )
        baseline = next(
            (row for row in cells if row["label"] == BASELINE and row["repeat"] == 1), None
        )
        if baseline is None:
            lines.append(f"## {model}\n\nno `{BASELINE}#1` cell to compare against\n")
            continue

        lines += [
            f"## {model}",
            "",
            f"Paired against `{BASELINE}#1` over {len(baseline['records'])} questions. "
            "`*` marks an interval excluding zero, which is not by itself a finding: "
            f"compare it with the `{BASELINE}` repeats.",
            "",
            "| cell | utility | recall | ungrounded | refused | Δ recall [95% CI] "
            "| lost/gained | Δ hit [95% CI] |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for row in cells:
            answers = row["answers"]
            name = f"{row['label']}#{row['repeat']}"
            if row is baseline:
                deltas = "— | — | —"
            else:
                recall = compare(baseline, row, "recall")
                hit = compare(baseline, row, "on_target")
                deltas = f"{_interval(recall)} | {recall.lost}/{recall.gained} | {_interval(hit)}"
            lines.append(
                f"| {name} | {answers['answer_utility']:.3f} | {answers['answer_recall']:.3f} "
                f"| {answers['ungrounded_id_rate']:.3f} | {answers['refusal_rate']:.3f} "
                f"| {deltas} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    print(report(load_rows(args.path)))


if __name__ == "__main__":
    main()
