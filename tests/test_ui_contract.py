"""The contract between the two static pages and the API they call.

The pages are plain HTML and JavaScript with no build step, so nothing checks
them. The failure this guards against is mundane and silent: an endpoint is
renamed or a response field is dropped, every Python test stays green, and a
page quietly stops working until someone opens it.

Two halves. The paths are *parsed out of the pages*, so a new fetch is covered
the moment it is written. The fields are *listed explicitly*, because the list
is the actual contract -- "the console cannot render an answer without
defenses_abstained" is a statement worth having to change deliberately.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from threatrag.api import main
from threatrag.api.main import app
from threatrag.domain.models import Answer, Chunk, RetrievedChunk
from threatrag.eval.findings import AttackCell, RetrieverCell, UtilityCell
from threatrag.security.attacks.runner import AttackResult

STATIC = Path(main.__file__).parent / "static"
PAGES = ("index.html", "findings.html")

# fetch("/api/x") and the console's post("/api/x", body) helper. A call whose
# path is a variable is deliberately not matched -- there is nothing to check.
CALL = re.compile(r'(?:fetch|post)\(\s*"(/[^"]*)"')
HREF = re.compile(r'href="(/[^"#]*)"')


def _page(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def _routes() -> set[str]:
    return {getattr(route, "path", "") for route in app.routes}


@pytest.mark.parametrize("page", PAGES)
def test_every_endpoint_a_page_calls_exists(page: str) -> None:
    called = set(CALL.findall(_page(page)))
    assert called, f"{page} calls no endpoints; the regex has probably stopped matching"
    missing = sorted(called - _routes())
    assert not missing, f"{page} calls endpoints that do not exist: {missing}"


@pytest.mark.parametrize("page", PAGES)
def test_every_internal_link_resolves(page: str) -> None:
    missing = sorted(set(HREF.findall(_page(page))) - _routes())
    assert not missing, f"{page} links to routes that do not exist: {missing}"


@pytest.mark.parametrize("path", ["/", "/findings"])
def test_pages_are_served(path: str) -> None:
    response = TestClient(app).get(path)
    assert response.status_code == 200
    assert "<html" in response.text


# What each page reads off the JSON. Dropping or renaming any of these breaks
# rendering without breaking a single other test.
REQUIRED: list[tuple[type[BaseModel], tuple[str, ...]]] = [
    # The console's answer panel, including the three outcomes it distinguishes.
    (
        Answer,
        (
            "text",
            "model",
            "citations",
            "unsupported_citations",
            "retrieved",
            "defenses_applied",
            "defenses_abstained",
            "stripped_urls",
            "blocked",
            "block_reason",
        ),
    ),
    (RetrievedChunk, ("chunk", "score")),
    # Both badges on a retrieved passage: who may read it, and how far it may
    # steer the answer.
    (Chunk, ("tlp", "trust_tier", "source_ref", "title", "text")),
    (main.DefenseView, ("name", "enabled_by_default")),
    (main.RetrieverView, ("name", "collection", "is_default")),
    (main.PrincipalView, ("id", "label")),
    (main.AttackView, ("id", "family", "description", "corpus", "target_query")),
    (main.AttackRunResponse, ("result", "defenses_applied", "retrieval")),
    (
        AttackResult,
        (
            "attack_id",
            "family",
            "succeeded",
            "criteria",
            "answer_text",
            "fired_urls",
            "defenses_abstained",
        ),
    ),
    # The findings page. landed_low/landed_high and unstable are the range the
    # page shows instead of an average.
    (
        AttackCell,
        ("defense_set", "model", "corpus", "total", "landed_low", "landed_high", "unstable"),
    ),
    (
        RetrieverCell,
        ("retrieval", "defenses", "corpus", "total", "landed_low", "landed_high", "unstable"),
    ),
    (UtilityCell, ("label", "model", "repeat", "utility", "recall", "warm")),
]


@pytest.mark.parametrize("model,fields", REQUIRED, ids=lambda v: getattr(v, "__name__", ""))
def test_response_fields_the_pages_read_still_exist(
    model: type[BaseModel], fields: tuple[str, ...]
) -> None:
    missing = sorted(set(fields) - set(model.model_fields))
    assert not missing, f"{model.__name__} no longer carries {missing}, which a page reads"


def test_the_console_still_renders_images() -> None:
    """The markdown image path is the surface M4's exfiltration attack targets.

    If a later tidy-up removes it, egress_filter has nothing to defend on
    screen and the attack demo silently becomes unfalsifiable -- it would look
    blocked whether or not the defence did anything.
    """
    assert "<img" in _page("index.html")


def test_abstention_is_worded_as_not_a_pass() -> None:
    """A defence that could not judge must not read as one that cleared."""
    assert "could not judge" in _page("index.html")
