"""API tests.

Services are injected through the dependency override rather than the lifespan,
so these run without Qdrant or Ollama. The routes are thin, and what is worth
testing about them is the access-control contract: which principal a request
runs as, and what happens when it names one that does not exist.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from threatrag.api.main import Services, app, get_services
from threatrag.config import Config, PrincipalConfig
from threatrag.domain.models import TLP, Answer, Principal
from threatrag.rag.generators.ollama import GenerationError


class StubPipeline:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.seen: Principal | None = None

    def answer(self, question: str, *, principal: Principal | None = None, k: int | None = None):
        self.seen = principal
        if self.error is not None:
            raise self.error
        return Answer(question=question, text="grounded [1]", model="stub", citations=["A — a"])

    def retrieve(self, question: str, *, principal: Principal | None = None, k: int | None = None):
        self.seen = principal
        return []


def _config() -> Config:
    return Config.model_validate(
        {
            "embedding": {
                "primary": "m",
                "models": {"m": {"model_id": "x", "dim": 3}},
            },
            "principals": {
                "analyst": {"label": "Analyst", "clearance": "clear"},
                "ir_lead": {"label": "IR lead", "role": "ir_lead", "clearance": "red"},
            },
        }
    )


@pytest.fixture
def client_and_pipeline() -> Any:
    pipeline = StubPipeline()

    def override() -> Services:
        return Services(config=_config(), pipeline=pipeline)  # type: ignore[arg-type]

    app.dependency_overrides[get_services] = override
    yield TestClient(app), pipeline
    app.dependency_overrides.clear()


def test_health() -> None:
    assert TestClient(app).get("/health").json()["status"] == "ok"


def test_principals_come_from_config(client_and_pipeline: Any) -> None:
    client, _ = client_and_pipeline
    body = client.get("/api/principals").json()
    assert [p["id"] for p in body] == ["analyst", "ir_lead"]
    assert body[1]["clearance"] == "red"


def test_ask_runs_as_the_named_principal(client_and_pipeline: Any) -> None:
    client, pipeline = client_and_pipeline
    response = client.post(
        "/api/ask", json={"question": "q"}, headers={"X-Principal-Id": "ir_lead"}
    )
    assert response.status_code == 200
    assert response.json()["citations"] == ["A — a"]
    assert pipeline.seen is not None
    assert pipeline.seen.clearance is TLP.RED


def test_unknown_principal_is_refused_not_defaulted(client_and_pipeline: Any) -> None:
    """Defaulting on a bad header is how an access-control bug goes silent."""
    client, pipeline = client_and_pipeline
    response = client.post("/api/ask", json={"question": "q"}, headers={"X-Principal-Id": "ghost"})
    assert response.status_code == 400
    assert pipeline.seen is None


def test_clearance_is_never_taken_from_the_request(client_and_pipeline: Any) -> None:
    """A caller may choose an identity; it may not choose that identity's clearance."""
    client, pipeline = client_and_pipeline
    client.post(
        "/api/ask",
        json={"question": "q", "clearance": "red"},
        headers={"X-Principal-Id": "analyst"},
    )
    assert pipeline.seen is not None
    assert pipeline.seen.clearance is TLP.CLEAR


def test_generation_failure_is_a_bad_gateway(client_and_pipeline: Any) -> None:
    client, pipeline = client_and_pipeline
    pipeline.error = GenerationError("ollama down")
    response = client.post("/api/ask", json={"question": "q"})
    assert response.status_code == 502
    assert "ollama down" in response.json()["detail"]


def test_empty_question_is_rejected(client_and_pipeline: Any) -> None:
    client, _ = client_and_pipeline
    assert client.post("/api/ask", json={"question": ""}).status_code == 422


def test_principal_config_defaults_to_clear() -> None:
    assert PrincipalConfig(label="x").clearance is TLP.CLEAR
