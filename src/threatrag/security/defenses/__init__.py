"""Independently toggleable mitigations.

Each defence implements the :class:`~threatrag.domain.ports.Defense` protocol
and does its work in exactly one hook, so a defence set composes as a plain
pipeline and M7 can attribute a change to one mitigation rather than to a
bundle.

**Order is fixed here, not by the config.** A benchmark cell names a *set* of
defences; if the order they were listed in changed the result, ``{a, b}`` and
``{b, a}`` would be different cells and the sweep would carry a hidden
variable. ``CANONICAL_ORDER`` is that order, and it follows the request's own
path -- ingest, then retrieval, then answer -- so a defence never runs before
the stage it inspects.
"""

from __future__ import annotations

from collections.abc import Callable

from threatrag.config import Config
from threatrag.domain.ports import Defense

# name -> constructor. The order is the order defences run in.
_REGISTRY: dict[str, Callable[[Config], Defense]] = {}

CANONICAL_ORDER: tuple[str, ...] = ()


def register(name: str, build: Callable[[Config], Defense]) -> None:
    _REGISTRY[name] = build
    global CANONICAL_ORDER
    CANONICAL_ORDER = tuple(_REGISTRY)


def _install() -> None:
    """Register every defence, in the order they run.

    Imported here rather than at module scope so the registry stays the single
    place the canonical order is written down.
    """
    from threatrag.security.defenses import injection_screen

    register("injection_screen", injection_screen.build)


def available() -> tuple[str, ...]:
    return CANONICAL_ORDER


def build_defenses(config: Config) -> list[Defense]:
    """Instantiate the enabled defences, in canonical order.

    An unknown name is an error rather than a silent skip: a benchmark cell
    that quietly ran undefended would be recorded as a defended result, and
    nothing downstream could tell the difference.
    """
    unknown = sorted(set(config.defenses) - set(_REGISTRY))
    if unknown:
        raise ValueError(f"Unknown defence(s) {unknown}; available: {list(_REGISTRY)}")
    enabled = set(config.defenses)
    return [_REGISTRY[name](config) for name in CANONICAL_ORDER if name in enabled]


_install()
