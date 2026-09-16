"""Paired utility comparison over a utility-only benchmark JSONL.

Every cell is compared, question by question, with *every* undefended repeat of
its own model, and the report gives the range over those comparisons. There is
no single baseline because the undefended repeats disagree with each other: at
temperature 0 on the same index, two runs of one configuration in one session
rewrote up to 72 of 200 answers, so whichever repeat was chosen as "the"
baseline would move the result. An undefended repeat goes through the same
comparison against the other repeats, and those rows are the noise floor.

A cell's change counts as a finding only when its interval excludes zero, on
the same side, against every repeat. Anything less is within what re-running
the undefended configuration already produces.

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


def _span(values: Sequence[float], fmt: str) -> str:
    low, high = min(values), max(values)
    return format(low, fmt) if low == high else f"{format(low, fmt)} to {format(high, fmt)}"


def _clears_every(deltas: Sequence[PairedDelta]) -> str:
    """Name the direction when every comparison excludes zero on one side."""
    if all(delta.low > 0 for delta in deltas):
        return "higher"
    if all(delta.high < 0 for delta in deltas):
        return "lower"
    return "no"


def _rewritten(baseline: dict[str, Any], treated: dict[str, Any]) -> int:
    base = {record["id"]: record["text"] for record in baseline["records"]}
    return sum(record["text"] != base[record["id"]] for record in treated["records"])


def report(rows: Sequence[dict[str, Any]]) -> str:
    lines: list[str] = []
    for model in sorted({row["model"] for row in rows}):
        cells = sorted(
            (row for row in rows if row["model"] == model),
            key=lambda row: (row["label"] != BASELINE, row["label"], row["repeat"]),
        )
        repeats = [row for row in cells if row["label"] == BASELINE]
        lines += [f"## {model}", ""]
        if len(repeats) < 2:
            lines.append(f"needs at least two `{BASELINE}` repeats to measure the noise floor\n")
            continue

        lines += [
            f"Each cell paired against every `{BASELINE}` repeat other than itself "
            f"({len(repeats)} repeats, {len(repeats[0]['records'])} questions); columns give "
            "the range over those comparisons. *Finding* is `higher`/`lower` only when the "
            "95% interval excludes zero on that side against every repeat, on hit or on recall.",
            "",
            "| cell | utility | recall | ungrounded | refused | Δ hit | Δ recall "
            "| recall lost/gained | answers rewritten | finding |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for row in cells:
            answers = row["answers"]
            others = [base for base in repeats if base is not row]
            hit = [compare(base, row, "on_target") for base in others]
            recall = [compare(base, row, "recall") for base in others]
            verdicts = {_clears_every(hit), _clears_every(recall)} - {"no"}
            finding = "/".join(sorted(verdicts)) or "no"
            lost_gained = sorted({f"{d.lost}/{d.gained}" for d in recall})
            lines.append(
                f"| {row['label']}#{row['repeat']} | {answers['answer_utility']:.3f} "
                f"| {answers['answer_recall']:.3f} | {answers['ungrounded_id_rate']:.3f} "
                f"| {answers['refusal_rate']:.3f} "
                f"| {_span([d.mean_delta for d in hit], '+.3f')} "
                f"| {_span([d.mean_delta for d in recall], '+.3f')} "
                f"| {', '.join(lost_gained)} "
                f"| {_span([_rewritten(base, row) for base in others], 'd')} "
                f"| {finding} |"
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
