"""The M7 sweep harness.

Two behaviours are load-bearing and both are about not losing or faking a run:
results reach disk per cell so an interrupted sweep resumes instead of being
restarted shorter, and a cell's config is a copy so one cell's defences cannot
leak into the next.
"""

from __future__ import annotations

from pathlib import Path

from threatrag.config import Config
from threatrag.eval.benchmark import (
    Cell,
    CellResult,
    ResultWriter,
    RouteResult,
    build_cells,
    cell_config,
    iter_cells,
    summarise_route,
)

DEFENCE_SETS: list[tuple[str, Path | None]] = [
    ("none", None),
    ("injection_screen", Path("configs/experiments/defense_injection_screen.yaml")),
    ("all", Path("configs/experiments/defense_all.yaml")),
]
MODELS = ["qwen2.5:7b", "qwen2.5:1.5b"]


class FakeResult:
    def __init__(self, attack_id: str, succeeded: bool) -> None:
        self.attack_id = attack_id
        self.succeeded = succeeded


def _base_config() -> Config:
    return Config(
        embedding={"primary": "e", "models": {"e": {"model_id": "m", "dim": 3}}},  # type: ignore[arg-type]
        defenses=["provenance_fence"],
    )


def _result(label: str, model: str) -> CellResult:
    return CellResult(
        label=label,
        defenses=[],
        model=model,
        routes=[RouteResult(corpus="m4", landed=1, total=7, survivors=["inj-003"])],
        answers={"answer_utility": 0.5},
        retrieval=None,
        seconds=1.0,
    )


def test_the_defence_set_varies_slowest() -> None:
    """A partial sweep should cover every defence set for one model, not half
    the defence sets for both -- only the first is readable as a result."""
    cells = build_cells(DEFENCE_SETS, MODELS)

    assert [c.key for c in cells[:4]] == [
        "none@qwen2.5:7b",
        "none@qwen2.5:1.5b",
        "injection_screen@qwen2.5:7b",
        "injection_screen@qwen2.5:1.5b",
    ]
    assert len(cells) == 6


def test_a_cell_loads_its_own_overlay_rather_than_sharing_an_object() -> None:
    """Nothing a previous cell set may survive into this one, and the config
    must be exactly what running that overlay from the command line produces."""
    seen: list[Path | None] = []

    def load(config_path: Path | None, overlay: Path | None) -> Config:
        seen.append(overlay)
        built = _base_config()
        built.defenses = ["injection_screen", "corroboration"]
        return built

    cell = Cell("all", Path("configs/experiments/defense_all.yaml"), "qwen2.5:1.5b")
    built = cell_config(cell, config_path=None, load=load)

    assert seen == [Path("configs/experiments/defense_all.yaml")]
    assert built.defenses == ["injection_screen", "corroboration"]
    assert built.generation.model == "qwen2.5:1.5b"


def test_the_source_cap_overlay_carries_its_overfetch() -> None:
    """The reason cells name an overlay instead of a defence list. Without the
    headroom the cap can only delete, and the cell would measure a different
    mitigation than the one M6 published under that name."""
    from threatrag.config import load_config

    cfg = load_config(overlay="configs/experiments/defense_source_cap.yaml")

    assert cfg.defenses == ["source_cap"]
    assert cfg.retrieval.overfetch == 3


def test_results_are_readable_after_every_cell(tmp_path: Path) -> None:
    writer = ResultWriter(tmp_path / "sweep.jsonl")

    writer.write(_result("none", "qwen2.5:7b"))
    writer.write(_result("none", "qwen2.5:1.5b"))

    assert writer.completed() == {"none@qwen2.5:7b", "none@qwen2.5:1.5b"}


def test_an_interrupted_sweep_resumes_where_it_stopped(tmp_path: Path) -> None:
    writer = ResultWriter(tmp_path / "sweep.jsonl")
    writer.write(_result("none", "qwen2.5:7b"))

    remaining = [c.key for c in iter_cells(build_cells(DEFENCE_SETS, MODELS), writer)]

    assert "none@qwen2.5:7b" not in remaining
    assert len(remaining) == 5


def test_a_fresh_sweep_runs_every_cell(tmp_path: Path) -> None:
    writer = ResultWriter(tmp_path / "absent.jsonl")

    assert len(list(iter_cells(build_cells(DEFENCE_SETS, MODELS), writer))) == 6


def test_a_route_records_which_attacks_survived_not_only_how_many() -> None:
    """A count cannot tell M7 whether two defence sets stop the same attacks or
    different ones, which is the whole orthogonality question."""
    route = summarise_route(
        "m4",
        [FakeResult("inj-003", True), FakeResult("poi-001", False), FakeResult("exf-002", True)],  # type: ignore[list-item]
    )

    assert route.landed == 2
    assert route.survivors == ["inj-003", "exf-002"]
    assert route.success_rate == 2 / 3


def test_an_empty_route_does_not_divide_by_zero() -> None:
    assert summarise_route("m4", []).success_rate == 0.0


def test_routes_survive_the_json_round_trip(tmp_path: Path) -> None:
    writer = ResultWriter(tmp_path / "sweep.jsonl")

    writer.write(_result("none", "qwen2.5:7b"))

    row = writer.path.read_text(encoding="utf-8").strip()
    assert '"survivors": ["inj-003"]' in row
    assert '"success_rate"' in row
