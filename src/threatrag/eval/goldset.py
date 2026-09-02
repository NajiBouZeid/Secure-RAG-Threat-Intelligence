"""Retrieval gold set derived from ATT&CK's own relationship graph.

The labels come from MITRE's curated ``intrusion-set --uses--> attack-pattern``
edges, not from an LLM asked to invent questions about the corpus. That
distinction is what makes Recall@k here a measurement rather than a
self-fulfilling one: an LLM-generated question set is written *from* the chunks
being retrieved, so it rewards the retriever for surfacing the text the question
was copied out of.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# A group/tactic pair with only one or two techniques makes Recall@5 trivially
# 1.0 and tells us nothing; too many and no top-k could ever cover it.
MIN_TECHNIQUES = 3
MAX_TECHNIQUES = 8


class GoldQuery(BaseModel):
    """One evaluation question and the source refs a correct retrieval must surface."""

    id: str
    question: str
    relevant: list[str]
    tags: dict[str, str] = Field(default_factory=dict)


class GoldSet(BaseModel):
    name: str
    queries: list[GoldQuery]

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> GoldSet:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing. Run `threatrag build-goldset` before evaluating."
            )
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _humanise(tactic: str) -> str:
    return tactic.replace("-", " ")


def build_from_attack(
    groups: dict[str, dict[str, Any]],
    *,
    limit: int = 200,
    seed: int = 1337,
) -> GoldSet:
    """Turn group-to-technique mappings into questions a SOC analyst would ask.

    Sampling is seeded so the gold set is byte-identical across machines --
    a benchmark whose question set drifts between runs cannot support the
    before/after defence comparisons Phase 3 depends on.
    """
    queries: list[GoldQuery] = []

    for group_id, entry in sorted(groups.items()):
        name = str(entry["name"])
        techniques: dict[str, list[str]] = entry["techniques"]

        for tactic, technique_ids in sorted(techniques.items()):
            if tactic == "unspecified":
                continue
            unique = sorted(set(technique_ids))
            if not MIN_TECHNIQUES <= len(unique) <= MAX_TECHNIQUES:
                continue
            queries.append(
                GoldQuery(
                    id=f"{group_id}:{tactic}",
                    question=f"Which {_humanise(tactic)} techniques does {name} use?",
                    relevant=unique,
                    tags={"group": group_id, "group_name": name, "tactic": tactic},
                )
            )

    rng = random.Random(seed)
    rng.shuffle(queries)
    selected = sorted(queries[:limit], key=lambda query: query.id)
    return GoldSet(name="attack-group-technique", queries=selected)


def gold_refs(queries: Sequence[GoldQuery]) -> set[str]:
    """Every source ref referenced by the gold set, for sanity-checking coverage."""
    return {ref for query in queries for ref in query.relevant}
