"""Does 1.5b out-score 7b on answer utility because it answers better, or
because it answers longer?

M7's utility metric asks only whether a relevant identifier appears anywhere in
the answer, so a model that writes more has more room to name one by accident.
qwen2.5:1.5b scores roughly double qwen2.5:7b and writes roughly double the
characters, which is exactly the shape a metric artefact would have.

This asks the 50 gold questions the sweep uses -- same seed, same sample, same
`principal=None`, undefended -- and records the length of every answer beside
whether it scored. The sweep's own hit predicate is imported rather than
reimplemented: a difference measured here must not be a difference in the
metric.

It is a script rather than a CLI command because it answers one question about
one milestone's numbers, and nothing in the product needs it. Roughly ten
minutes on an RTX 4060 with both models already pulled:

    ./.venv/Scripts/python.exe scripts/answer_length.py

Writes reports/data/m7_answer_length.json, which reports/m7_benchmark.md cites.
"""

from __future__ import annotations

import json
import random
import statistics as stats
from pathlib import Path
from typing import Any

from threatrag import factory
from threatrag.config import load_config
from threatrag.eval import goldset as goldset_module
from threatrag.eval.answers import names_a_relevant_technique
from threatrag.rag.pipeline import NO_EVIDENCE

# Both fixed to the sweep's values. Changing either makes the numbers here
# incomparable with the utility column in reports/m7_benchmark.md.
SEED = 1337
QUESTIONS = 50
GOLDSET_FILENAME = "attack_goldset.json"
MODELS = ("qwen2.5:7b", "qwen2.5:1.5b")
OUT = Path("reports/data/m7_answer_length.json")


def summarise(model: str, records: list[dict[str, Any]]) -> None:
    hits = [r for r in records if r["hit"]]
    misses = [r for r in records if not r["hit"]]
    chars = [int(r["chars"]) for r in records]
    print(model)
    print(
        f"  utility: {len(hits) / len(records):.3f}"
        f"  refusals: {sum(1 for r in records if r['refused'])}"
    )
    print(f"  mean chars: {stats.mean(chars):.0f}  median: {stats.median(chars):.0f}")
    # The direction that matters. If length were buying hits, the hits would be
    # the longer group.
    if hits:
        print(f"  mean chars HIT:  {stats.mean(int(r['chars']) for r in hits):.0f}")
    if misses:
        print(f"  mean chars MISS: {stats.mean(int(r['chars']) for r in misses):.0f}")

    ranked = sorted(records, key=lambda r: int(r["chars"]))
    third = len(ranked) // 3
    if third == 0:
        # Fewer than three answers leaves an empty tercile. Only reachable by
        # lowering QUESTIONS, but a crash while summarising is a poor way to
        # find that out after the generation has already been paid for.
        print("  too few answers to split into terciles")
        return
    for label, part in (
        ("short", ranked[:third]),
        ("mid", ranked[third : 2 * third]),
        ("long", ranked[2 * third :]),
    ):
        scored = sum(1 for r in part if r["hit"])
        print(f"  {label:<5} n={len(part):<3} hit rate {scored / len(part):.2f}")


def main() -> None:
    base = load_config(None)
    gold = goldset_module.GoldSet.load(base.paths.eval_dir / GOLDSET_FILENAME)
    sample = random.Random(SEED).sample(list(gold.queries), QUESTIONS)

    everything: dict[str, list[dict[str, Any]]] = {}
    for model in MODELS:
        # Loaded fresh per model, for the reason cell_config does the same: so
        # nothing the previous model set can survive into this one.
        config = load_config(None)
        config.generation.model = model
        pipeline = factory.build_answer_pipeline(config)

        records: list[dict[str, Any]] = []
        for query in sample:
            answer = pipeline.answer(query.question)
            records.append(
                {
                    "question": query.question,
                    "chars": len(answer.text),
                    "hit": names_a_relevant_technique(answer, query.relevant),
                    "refused": answer.blocked or answer.text.strip() == NO_EVIDENCE,
                }
            )
        everything[model] = records
        summarise(model, records)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" so regenerating the artefact does not show up as a whole-file
    # diff on Windows.
    with OUT.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(everything, handle, indent=2)
        handle.write("\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
