"""Defence-set composition.

What is under test is the toggle mechanism rather than any one mitigation: that
a set of defences composes in a fixed order, that the order a config happens to
list them in cannot change a result, and that a name nobody implements is an
error instead of a silently undefended run.
"""

from __future__ import annotations

import pytest

from threatrag.config import Config
from threatrag.domain.models import Answer
from threatrag.domain.ports import Defense
from threatrag.security import defenses as reg


class Marker:
    """A defence that records the order it ran in on the answer text."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def on_answer(self, answer: Answer) -> Answer:
        return answer.model_copy(update={"text": f"{answer.text}{self._name}"})


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reg, "_REGISTRY", {})
    monkeypatch.setattr(reg, "CANONICAL_ORDER", ())
    for name in ("alpha", "beta", "gamma"):
        reg.register(name, lambda _cfg, n=name: Marker(n))  # type: ignore[misc]


def _config(names: list[str]) -> Config:
    return Config(
        embedding={"primary": "e", "models": {"e": {"model_id": "m", "dim": 3}}},  # type: ignore[arg-type]
        defenses=names,
    )


def _apply(built: list[Defense]) -> str:
    answer = Answer(question="q", text="")
    for defense in built:
        answer = defense.on_answer(answer)
    return answer.text


def test_only_the_named_defences_are_built(registry: None) -> None:
    assert [d.name for d in reg.build_defenses(_config(["beta"]))] == ["beta"]


def test_an_empty_set_is_the_undefended_baseline(registry: None) -> None:
    assert reg.build_defenses(_config([])) == []


def test_config_order_cannot_change_the_result(registry: None) -> None:
    """A benchmark cell names a set. If {a,b} and {b,a} differed, the sweep
    would carry an unrecorded variable."""
    forward = _apply(reg.build_defenses(_config(["alpha", "gamma"])))
    reverse = _apply(reg.build_defenses(_config(["gamma", "alpha"])))

    assert forward == reverse == "alphagamma"


def test_an_unknown_defence_is_refused_not_skipped(registry: None) -> None:
    """A skipped name would record an undefended run as a defended one."""
    with pytest.raises(ValueError, match="ghost"):
        reg.build_defenses(_config(["alpha", "ghost"]))
