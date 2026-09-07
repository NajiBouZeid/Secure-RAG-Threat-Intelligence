"""The M7 sweep: defence sets x models, on both attack routes and three utility axes.

The structure is forced by what M5 and M6 found, not chosen for tidiness.

*Two attack routes, never summed.* M4's attacks arrive through the prompt and
M5's read stored vectors without issuing a query, so a defence set that stops
one may do nothing about the other. Collapsing them into a single "attack
success rate" would let a defence average its way to a good number while leaving
a whole route untouched. This runner measures the prompt route; the index route
is measured separately, because it needs its own index rather than its own cell.

*Retrieval is scored once per defence set, not once per cell.* Nothing in the
retrieval path depends on the generator, so pairing it with a model would double
the cost and invent a difference between two identical numbers.

*Results stream to disk as they are produced.* A full sweep is hours of
generation, and a crash in the last cell must not cost the first fifteen.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from threatrag.config import Config
from threatrag.domain.models import Answer, Principal
from threatrag.eval.answers import AnswerScores, score_answers
from threatrag.eval.goldset import GoldQuery
from threatrag.eval.metrics import AggregateScores, aggregate, score_query
from threatrag.security.attacks.runner import AttackResult
from threatrag.security.attacks.schema import Attack


@dataclass(frozen=True)
class Cell:
    """One benchmark configuration: a defence set under one generator."""

    label: str
    defenses: tuple[str, ...]
    model: str

    @property
    def key(self) -> str:
        return f"{self.label}@{self.model}"


@dataclass
class RouteResult:
    """Outcome of one attack corpus against one cell."""

    corpus: str
    landed: int
    total: int
    survivors: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.landed / self.total if self.total else 0.0


@dataclass
class CellResult:
    label: str
    defenses: list[str]
    model: str
    routes: list[RouteResult]
    answers: dict[str, float | int]
    retrieval: dict[str, float | int] | None
    seconds: float

    def as_json(self) -> dict[str, object]:
        payload = asdict(self)
        payload["routes"] = [
            {**asdict(route), "success_rate": round(route.success_rate, 4)} for route in self.routes
        ]
        payload["seconds"] = round(self.seconds, 1)
        return payload


def score_retrieval(
    retrieve: Callable[[str, int], Sequence[str]], queries: Sequence[GoldQuery], k: int
) -> AggregateScores:
    """Retrieval quality over the gold set.

    ``retrieve`` returns source_refs in rank order. Duplicates are collapsed
    before scoring, so several chunks of one technique count as one hit rather
    than inflating recall -- the same rule ``eval-retrieval`` uses, kept
    identical here so a sweep row is comparable with an M1 number.
    """
    scores = []
    for item in queries:
        seen: list[str] = []
        for ref in retrieve(item.question, k):
            if ref not in seen:
                seen.append(ref)
        scores.append(score_query(seen, item.relevant, k))
    return aggregate(scores)


def answer_gold_questions(
    answer: Callable[[str, Principal | None], Answer],
    queries: Sequence[GoldQuery],
    *,
    principal: Principal | None = None,
    no_evidence: str,
) -> AnswerScores:
    return score_answers(
        [answer(item.question, principal) for item in queries], queries, no_evidence=no_evidence
    )


def summarise_route(corpus: str, results: Sequence[AttackResult]) -> RouteResult:
    return RouteResult(
        corpus=corpus,
        landed=sum(1 for r in results if r.succeeded),
        total=len(results),
        survivors=[r.attack_id for r in results if r.succeeded],
    )


class ResultWriter:
    """Append-only JSONL sink, flushed per cell.

    A sweep is hours long. Holding results in memory until the end means a
    crash, an OOM or a killed terminal discards every cell that already ran, and
    the temptation then is to re-run a shortened sweep and quietly compare it
    against numbers from the longer one.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def completed(self) -> set[str]:
        """Cell keys already on disk, so an interrupted sweep can resume."""
        if not self._path.exists():
            return set()
        done = set()
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done.add(f"{row['label']}@{row['model']}")
        return done

    def write(self, result: CellResult) -> None:
        with self._path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(result.as_json(), ensure_ascii=False) + "\n")


def build_cells(
    defence_sets: Sequence[tuple[str, tuple[str, ...]]], models: Sequence[str]
) -> list[Cell]:
    """Every defence set under every model, defence set varying slowest.

    Slowest on purpose: a partial sweep then covers every defence set for the
    first model rather than half the defence sets for both, which is the half
    that can still be read as a result.
    """
    return [
        Cell(label=label, defenses=names, model=model)
        for label, names in defence_sets
        for model in models
    ]


def cell_config(base: Config, cell: Cell) -> Config:
    """Base config with this cell's defences and generator applied.

    A deep copy: the defence list and the model are the only things a cell is
    allowed to vary, and mutating the shared base would leak one cell's settings
    into the next.
    """
    config = base.model_copy(deep=True)
    config.defenses = list(cell.defenses)
    config.generation.model = cell.model
    return config


def iter_cells(cells: Sequence[Cell], writer: ResultWriter) -> Iterator[Cell]:
    """Yield the cells not already recorded, skipping the rest."""
    done = writer.completed()
    for cell in cells:
        if cell.key not in done:
            yield cell


def timed(start: float) -> float:
    return time.perf_counter() - start


def run_attacks(
    runner_run: Callable[[Attack], AttackResult], attacks: Sequence[Attack], corpus: str
) -> RouteResult:
    return summarise_route(corpus, [runner_run(attack) for attack in attacks])
