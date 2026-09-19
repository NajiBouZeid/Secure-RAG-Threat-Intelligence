"""API tests.

Services are injected through the dependency override rather than the lifespan,
so these run without Qdrant or Ollama. The routes are thin, and what is worth
testing about them is the access-control contract: which principal a request
runs as, and what happens when it names one that does not exist.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from threatrag.api import main
from threatrag.api.main import (
    Services,
    app,
    config_for,
    config_with_defenses,
    get_services,
)
from threatrag.config import Config, PrincipalConfig
from threatrag.domain.models import TLP, Answer, Principal
from threatrag.rag.generators.ollama import GenerationError
from threatrag.security.defenses import available as available_defenses


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


class StubServices:
    """Stands in for Services without building an embedder or a store.

    Records the defence set each request asked for, which is the part of the
    new wiring worth asserting: the routes must pass the *requested* set
    through rather than serving whatever the server started with.
    """

    def __init__(self, config: Config, pipeline: StubPipeline) -> None:
        self.config = config
        self.pipeline = pipeline
        self.requested: list[str] | None = None
        self.retrieval: str | None = None

    def pipeline_for(self, defenses: Any, retrieval: Any = None) -> StubPipeline:
        self.requested = list(defenses)
        self.retrieval = retrieval
        return self.pipeline


@pytest.fixture
def client_and_pipeline() -> Any:
    pipeline = StubPipeline()
    services = StubServices(_config(), pipeline)

    def override() -> Services:
        return services  # type: ignore[return-value]

    app.dependency_overrides[get_services] = override
    client = TestClient(app)
    client.services = services  # type: ignore[attr-defined]
    yield client, pipeline
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


def test_defenses_are_listed_in_canonical_order(client_and_pipeline: Any) -> None:
    """The order a set runs in belongs to the registry, not to the client."""
    client, _ = client_and_pipeline
    body = client.get("/api/defenses").json()
    assert [d["name"] for d in body] == list(available_defenses())
    assert all(d["enabled_by_default"] is False for d in body)


def test_request_defences_reach_the_pipeline(client_and_pipeline: Any) -> None:
    client, _ = client_and_pipeline
    client.post("/api/ask", json={"question": "q", "defenses": ["egress_filter"]})
    assert client.services.requested == ["egress_filter"]  # type: ignore[attr-defined]


def test_omitted_and_empty_defences_are_different(client_and_pipeline: Any) -> None:
    """None means "the server's configuration"; [] means "explicitly none".

    Collapsing them would make an undefended request indistinguishable from a
    default one, which is precisely the comparison this UI exists to show.
    """
    client, _ = client_and_pipeline
    config = _config()
    config.defenses = ["egress_filter"]
    client.services.config = config  # type: ignore[attr-defined]

    client.post("/api/ask", json={"question": "q"})
    assert client.services.requested == ["egress_filter"]  # type: ignore[attr-defined]

    client.post("/api/ask", json={"question": "q", "defenses": []})
    assert client.services.requested == []  # type: ignore[attr-defined]


def test_unknown_defence_is_refused() -> None:
    with pytest.raises(HTTPException) as caught:
        config_with_defenses(_config(), ["not_a_defence"])
    assert caught.value.status_code == 400
    assert "not_a_defence" in caught.value.detail


def test_search_does_not_report_what_a_clearance_cannot_see(client_and_pipeline: Any) -> None:
    """A withheld count would need a query at a clearance the caller lacks."""
    client, _ = client_and_pipeline
    body = client.post("/api/search", json={"question": "q"}).json()
    assert isinstance(body, list)


def test_hybrid_plus_segregation_is_refused_with_the_config_message() -> None:
    """The combination rule lives in the config validator; the API must not
    duplicate it, and must not skip it by copying the model instead."""
    config = _config()
    config.retrieval.mode = "hybrid"
    config.retrieval.score_threshold = None
    with pytest.raises(HTTPException) as caught:
        config_with_defenses(config, ["corpus_segregation"])
    assert caught.value.status_code == 400
    assert "rank, not a similarity" in caught.value.detail


def test_unknown_retriever_is_refused() -> None:
    with pytest.raises(HTTPException) as caught:
        config_for("not_a_retriever", [])
    assert caught.value.status_code == 400
    assert "not_a_retriever" in caught.value.detail


def test_named_retrievers_load_their_committed_overlay() -> None:
    """What the UI calls "hybrid" must be the configuration that was measured."""
    assert config_for("dense", []).retrieval.mode == "dense"
    hybrid = config_for("hybrid", [])
    assert hybrid.retrieval.mode == "hybrid"
    assert hybrid.vector_store.collection == "threatrag_hybrid"
    assert config_for("hybrid_text", []).vector_store.collection == "threatrag_hybrid_text"


def test_requested_retriever_reaches_the_pipeline(client_and_pipeline: Any) -> None:
    client, _ = client_and_pipeline
    client.post("/api/ask", json={"question": "q", "retrieval": "hybrid"})
    assert client.services.retrieval == "hybrid"  # type: ignore[attr-defined]


def test_attack_list_does_not_hand_out_payloads(client_and_pipeline: Any) -> None:
    """The poison text is in the repo; an endpoint that serves it turns a
    running instance into a source of working injection strings."""
    client, _ = client_and_pipeline
    body = client.get("/api/attacks").json()
    assert body, "the committed attack corpus should not be empty"
    for attack in body:
        assert set(attack) == {
            "id",
            "family",
            "description",
            "corpus",
            "target_query",
            "tlp",
            "trust_tier",
        }


def test_unknown_attack_is_a_404(client_and_pipeline: Any) -> None:
    client, _ = client_and_pipeline
    response = client.post("/api/attacks/run", json={"attack_id": "no-such-attack"})
    assert response.status_code == 404


def test_only_one_attack_runs_at_a_time(client_and_pipeline: Any) -> None:
    """Two overlapping runs would each retrieve the other's poison."""
    client, _ = client_and_pipeline
    known = client.get("/api/attacks").json()[0]["id"]
    main.ATTACK_LOCK.acquire()
    try:
        response = client.post("/api/attacks/run", json={"attack_id": known})
    finally:
        main.ATTACK_LOCK.release()
    assert response.status_code == 409
