"""Dense against hybrid retrieval, all arms in one session.

Three arms over the same 40815 chunks: the dense control (`threatrag`), hybrid
with titles indexed (`threatrag_hybrid`) and hybrid over chunk text only
(`threatrag_hybrid_text`). The hybrid collections are copies of the control
with BM25 weights added, so their dense vectors are the control's own.

Two question sets, reported separately and never merged:

* the 200-question ATT&CK gold set, the headline every milestone reports;
* identifier probes ("What is T1055.001?") with name-phrased twins. They favour
  a lexical retriever by construction, which is why they are their own axis.

The dense arm is re-measured here rather than taken from a report, because a
retrieval number is only reproducible against a stated index state, and
today's dense recall has already drifted from M7's published figure. Every arm
is run twice and must agree with itself exactly before anything is compared.

    python scripts/hybrid_eval.py

Writes reports/data/hybrid_retrieval.json. A few minutes; no generator involved.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from threatrag import factory
from threatrag.config import Config, load_config
from threatrag.eval.goldset import GoldQuery, GoldSet
from threatrag.eval.metrics import aggregate, score_query
from threatrag.eval.paired import paired_delta
from threatrag.eval.probes import KINDS, build_identifier_probes
from threatrag.index.qdrant_store import QdrantVectorStore

GOLDSET = Path("data/eval/attack_goldset.json")
OUT = Path("reports/data/hybrid_retrieval.json")
ARMS: tuple[tuple[str, Path | None], ...] = (
    ("dense", None),
    ("hybrid", Path("configs/experiments/retrieval_hybrid.yaml")),
    ("hybrid_text", Path("configs/experiments/retrieval_hybrid_text.yaml")),
)
PROBES_PER_KIND = 150
SEED = 1337


def _documents(config: Config) -> list[tuple[str, str]]:
    store = QdrantVectorStore(config.vector_store.url, config.vector_store.collection)
    return sorted({(chunk.source_ref, chunk.title) for chunk in store.scroll_chunks()})


def _ranked(
    retrieve: Callable[[str], Sequence[str]], queries: Sequence[GoldQuery]
) -> list[list[str]]:
    out = []
    for item in queries:
        seen: list[str] = []
        for ref in retrieve(item.question):
            if ref not in seen:
                seen.append(ref)
        out.append(seen)
    return out


def _run_arm(config: Config, queries: Sequence[GoldQuery]) -> list[list[str]]:
    retriever = factory.build_retriever(config)
    k = config.retrieval.top_k

    def refs(question: str) -> list[str]:
        return [hit.chunk.source_ref for hit in retriever.retrieve(question, k=k)]

    return _ranked(refs, queries)


def _summary(ranked: Sequence[list[str]], queries: Sequence[GoldQuery], k: int) -> dict[str, Any]:
    scores = [score_query(r, q.relevant, k) for r, q in zip(ranked, queries, strict=True)]
    return dict(aggregate(scores).as_row())


def _value(ranked: list[str], query: GoldQuery, k: int, field: str) -> float:
    scores = score_query(ranked, query.relevant, k)
    return float(scores.hit) if field == "hit" else float(getattr(scores, field))


def _paired(
    base: Sequence[list[str]], treated: Sequence[list[str]], queries: Sequence[GoldQuery], k: int
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for field in ("hit", "reciprocal_rank", "recall_at_k"):
        delta = paired_delta(
            [_value(r, q, k, field) for r, q in zip(base, queries, strict=True)],
            [_value(r, q, k, field) for r, q in zip(treated, queries, strict=True)],
        )
        rows[field] = delta.as_row()
    return rows


def main() -> None:
    base = load_config()
    k = base.retrieval.top_k
    gold = GoldSet.load(GOLDSET)
    probes = build_identifier_probes(_documents(base), per_kind=PROBES_PER_KIND, seed=SEED)
    sets: dict[str, list[GoldQuery]] = {"gold": list(gold.queries)}
    for kind in KINDS:
        for phrasing in ("id", "name"):
            chosen = [
                q
                for q in probes.queries
                if q.tags["kind"] == kind and q.tags["phrasing"] == phrasing
            ]
            if chosen:
                sets[f"probe_{kind}_{phrasing}"] = chosen
    every = [q for queries in sets.values() for q in queries]

    ranked: dict[str, list[list[str]]] = {}
    seconds: dict[str, float] = {}
    for arm, overlay in ARMS:
        config = load_config(overlay=overlay)
        started = time.perf_counter()
        first = _run_arm(config, every)
        second = _run_arm(config, every)
        seconds[arm] = round(time.perf_counter() - started, 1)
        differing = sum(a != b for a, b in zip(first, second, strict=True))
        if differing:
            raise SystemExit(f"{arm}: {differing} queries ranked differently on a repeat run")
        ranked[arm] = first
        print(f"{arm}: {len(every)} queries x2, identical, {seconds[arm]} s", flush=True)

    result: dict[str, Any] = {
        "k": k,
        "probes_per_kind": PROBES_PER_KIND,
        "seed": SEED,
        "seconds": seconds,
        "sets": {},
    }
    offset = 0
    for name, queries in sets.items():
        span = slice(offset, offset + len(queries))
        offset += len(queries)
        entry: dict[str, Any] = {"queries": len(queries), "arms": {}, "vs_dense": {}}
        for arm, _ in ARMS:
            entry["arms"][arm] = _summary(ranked[arm][span], queries, k)
        for arm, _ in ARMS[1:]:
            entry["vs_dense"][arm] = _paired(ranked["dense"][span], ranked[arm][span], queries, k)
        entry["per_query"] = [
            {"id": q.id, "question": q.question, "relevant": q.relevant}
            | {arm: ranked[arm][span][i] for arm, _ in ARMS}
            for i, q in enumerate(queries)
        ]
        result["sets"][name] = entry

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(
        "\n| set | n | dense hit / MRR | hybrid hit / MRR | hybrid_text hit / MRR "
        "| Δhit hybrid [CI] | lost/gained |"
    )
    print("|---|---|---|---|---|---|---|")
    for name, entry in result["sets"].items():
        arms = entry["arms"]
        cells = " | ".join(f"{arms[a]['hit_rate']:.3f} / {arms[a]['mrr']:.3f}" for a, _ in ARMS)
        hit = entry["vs_dense"]["hybrid"]["hit"]
        print(
            f"| {name} | {entry['queries']} | {cells} | {hit['mean_delta']:+.3f} "
            f"[{hit['low']:+.3f}, {hit['high']:+.3f}] | {hit['lost']}/{hit['gained']} |"
        )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
