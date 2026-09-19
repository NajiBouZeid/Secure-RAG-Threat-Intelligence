"""FastAPI surface.

Thin over the library: every route resolves a principal, calls into the same
pipeline the CLI uses, and returns domain models directly. Nothing here decides
anything the benchmark would need to reproduce.

The principal arrives in a header the client chooses. That is deliberate and
documented rather than overlooked -- see PrincipalConfig. The interesting
property is that the API cannot widen access even if the header lies about a
role, because clearance comes from the server-side config and the store filters
on it inside the query.
"""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from threatrag import __version__, factory
from threatrag.config import Config, load_config
from threatrag.domain.models import Answer, Principal, RetrievedChunk
from threatrag.domain.ports import Embedder, Generator, VectorStore
from threatrag.eval import findings as findings_module
from threatrag.rag.generators.ollama import GenerationError
from threatrag.rag.pipeline import AnswerPipeline
from threatrag.security.attacks.loader import DEFAULT_ATTACK_DIR, load_attacks
from threatrag.security.attacks.runner import AttackResult
from threatrag.security.attacks.schema import Attack
from threatrag.security.attacks.sink import ExfiltrationSink
from threatrag.security.defenses import available as available_defenses

STATIC_DIR = Path(__file__).parent / "static"

PRINCIPAL_HEADER = "X-Principal-Id"

# Named retrievers, defined by the same committed overlays the benchmark ran,
# so what the UI calls "hybrid" is the configuration that was measured rather
# than a second copy of those settings that can drift from it.
# One attack at a time. A run indexes its poison document into the live
# collection and deletes it again, so two overlapping runs would each see the
# other's poison and neither result would mean anything.
ATTACK_LOCK = threading.Lock()

# The corpora an attack may be chosen from. Fixed, because the endpoint takes
# an id and never a payload: this lab runs the attacks it has under version
# control, and is not a service for running text someone posts to it.
ATTACK_DIRS: dict[str, Path] = {
    "m4": Path(DEFAULT_ATTACK_DIR),
    "evasion": Path(DEFAULT_ATTACK_DIR) / "evasion",
}

RETRIEVERS: dict[str, Path | None] = {
    "dense": None,
    "hybrid": Path("configs/experiments/retrieval_hybrid.yaml"),
    "hybrid_text": Path("configs/experiments/retrieval_hybrid_text.yaml"),
}


@dataclass(slots=True)
class Services:
    """The expensive, defence-independent parts of the system, built once.

    A request may ask for any defence set, and each set needs its own pipeline.
    Only the embedder, the stores and the generator are costly to build, so
    they are held here and the per-set wiring is done per request. Pipelines
    are deliberately *not* cached: assembling one is a handful of object
    constructions, and a cache keyed by defence set would be a second place
    where a stale configuration could survive a request.

    ``stores`` is keyed by whether ``corpus_segregation`` is enabled, because
    that defence chooses a different store rather than running a hook.
    """

    config: Config
    embedder: Embedder
    generator: Generator
    stores: dict[tuple[str, bool], VectorStore]

    def pipeline_for(self, defenses: Sequence[str], retrieval: str | None = None) -> AnswerPipeline:
        config = config_for(retrieval, defenses)
        # Keyed by both, because each names a different collection: the
        # retriever chooses which one holds the vectors, and segregation
        # chooses whether a second restricted one is merged in.
        key = (retrieval or "dense", "corpus_segregation" in config.defenses)
        store = self.stores.get(key)
        if store is None:
            store = self.stores.setdefault(key, factory.build_store(config))
        return factory.compose_answer_pipeline(
            config, embedder=self.embedder, store=store, generator=self.generator
        )


def config_for(retrieval: str | None, defenses: Sequence[str]) -> Config:
    """The config for one retriever and one defence set, built from disk.

    Reloaded rather than patched so a named retriever means exactly what its
    committed overlay says, including the settings -- BM25 average length,
    which collection -- that a hand-written patch here would have to restate.
    """
    if retrieval is not None and retrieval not in RETRIEVERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown retriever {retrieval!r}; available: {sorted(RETRIEVERS)}",
        )
    overlay = RETRIEVERS.get(retrieval or "dense")
    try:
        config = load_config(None, [overlay] if overlay is not None else None)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return config_with_defenses(config, defenses)


def config_with_defenses(config: Config, defenses: Sequence[str]) -> Config:
    """Re-derive the config with this defence set, through validation.

    Revalidated rather than copied because the combination rules live in the
    config's own validators -- hybrid retrieval plus ``corpus_segregation`` is
    refused there, and a copy would skip the check and quietly build something
    the CLI would not. Errors are surfaced to the caller as 400s carrying the
    config layer's own message.
    """
    unknown = sorted(set(defenses) - set(available_defenses()))
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown defence(s) {unknown}; available: {list(available_defenses())}",
        )
    try:
        return Config.model_validate({**config.model_dump(), "defenses": list(defenses)})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Built once at startup: the pipeline holds an embedder whose model would
    # otherwise be loaded from disk on every request.
    config = load_config()
    app.state.services = Services(
        config=config,
        embedder=factory.build_embedder(config),
        generator=factory.build_generator(config),
        stores={},
    )
    yield


app = FastAPI(title="Secure Threat Intelligence RAG", version=__version__, lifespan=lifespan)


def get_services(request: Request) -> Services:
    services: Services = request.app.state.services
    return services


ServicesDep = Annotated[Services, Depends(get_services)]


def resolve_principal(
    services: ServicesDep,
    x_principal_id: Annotated[str | None, Header(alias=PRINCIPAL_HEADER)] = None,
) -> Principal:
    """Turn the requested principal id into a Principal, or refuse.

    An unknown id is rejected rather than defaulted. Falling back to a default
    identity on a bad header is how an access-control bug becomes silent.
    """
    principals = services.config.principals
    if not principals:
        raise HTTPException(status_code=500, detail="No principals configured.")

    key = x_principal_id or next(iter(principals))
    spec = principals.get(key)
    if spec is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown principal {key!r}; configured: {sorted(principals)}",
        )
    return Principal(id=key, role=spec.role, clearance=spec.clearance)


PrincipalDep = Annotated[Principal, Depends(resolve_principal)]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int | None = Field(default=None, ge=1, le=50)
    # None means "whatever the server was configured with", which keeps the
    # request shape the CLI-equivalent default. An empty list is a different
    # thing and means explicitly undefended -- the two must not collapse.
    defenses: list[str] | None = Field(default=None)
    # None means the server's own configuration, as with defenses.
    retrieval: str | None = Field(default=None)


class DefenseView(BaseModel):
    name: str
    enabled_by_default: bool


class AttackView(BaseModel):
    id: str
    family: str
    description: str
    corpus: str
    target_query: str
    tlp: str
    trust_tier: int


class AttackRunRequest(BaseModel):
    # An id from the committed corpus, never attack text. The difference is
    # the difference between a lab that runs its own fixtures and a service
    # that will index whatever is posted to it.
    attack_id: str
    defenses: list[str] | None = Field(default=None)
    retrieval: str | None = Field(default=None)


class AttackRunResponse(BaseModel):
    result: AttackResult
    defenses_applied: list[str]
    retrieval: str


class RetrieverView(BaseModel):
    name: str
    mode: str
    collection: str
    is_default: bool


class PrincipalView(BaseModel):
    id: str
    label: str
    role: str
    clearance: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


def _reachable(url: str, path: str) -> str:
    """Liveness for one backing service, as a word the UI can print.

    Short timeout on purpose: this is called to tell someone their demo is
    about to fail, so it must answer faster than they can click again.
    """
    try:
        response = httpx.get(f"{url.rstrip('/')}{path}", timeout=2.0)
    except httpx.HTTPError:
        return "unreachable"
    return "ok" if response.is_success else f"http {response.status_code}"


@app.get("/api/health/services")
def service_health(services: ServicesDep) -> dict[str, str]:
    """Whether the two out-of-process dependencies are up.

    Worth its own route rather than folding into /health: /health answers
    "is this app running", which is true even when every question would fail.
    """
    config = services.config
    return {
        "qdrant": _reachable(config.vector_store.url, "/collections"),
        "ollama": _reachable(config.generation.url, "/api/tags"),
        "model": config.generation.model,
        "collection": config.vector_store.collection,
        "retrieval_mode": config.retrieval.mode,
    }


@app.get("/api/retrievers")
def retrievers(services: ServicesDep) -> list[RetrieverView]:
    """The retrievers the UI offers, and which one the server defaults to.

    Worth exposing because dense and hybrid fail at different things: dense
    answers 5-19% of identifier lookups and hybrid 83-100%, so a demo on the
    default retriever can look broken while working exactly as measured.

    Each one's mode and collection are read from its own overlay rather than
    described here, so this list cannot drift from what the retriever does.
    """
    current = (services.config.retrieval.mode, services.config.vector_store.collection)
    views = []
    for name in RETRIEVERS:
        config = config_for(name, [])
        identity = (config.retrieval.mode, config.vector_store.collection)
        views.append(
            RetrieverView(
                name=name,
                mode=config.retrieval.mode,
                collection=config.vector_store.collection,
                is_default=identity == current,
            )
        )
    return views


@app.get("/api/defenses")
def defenses(services: ServicesDep) -> list[DefenseView]:
    """The defences that can be toggled, in the order they run.

    The order is the registry's canonical one, not the caller's: a set of
    defences means the same thing however the client lists it.
    """
    configured = set(services.config.defenses)
    return [
        DefenseView(name=name, enabled_by_default=name in configured)
        for name in available_defenses()
    ]


@app.get("/api/principals")
def principals(services: ServicesDep) -> list[PrincipalView]:
    """The identities the UI offers. Labels come from config, not the client."""
    return [
        PrincipalView(id=key, label=spec.label, role=spec.role, clearance=spec.clearance.value)
        for key, spec in services.config.principals.items()
    ]


def _pipeline(services: Services, body: AskRequest) -> AnswerPipeline:
    requested = services.config.defenses if body.defenses is None else body.defenses
    return services.pipeline_for(requested, body.retrieval)


@app.post("/api/search")
def search(
    body: AskRequest, services: ServicesDep, principal: PrincipalDep
) -> list[RetrievedChunk]:
    """Retrieval only. Useful for seeing what a clearance can reach without
    paying for generation, and for telling a retrieval failure apart from a
    generation one.

    Each chunk carries its own ``tlp`` and ``trust_tier``, so what a clearance
    can reach is visible by asking as that principal -- not by asking the
    server what it is hiding. There is deliberately no "withheld" count here:
    producing one would mean running a second retrieval at a clearance the
    caller does not hold, and an API that will do that on request is a worse
    thing to ship than a demo that shows one less number.
    """
    return _pipeline(services, body).retrieve(body.question, principal=principal, k=body.k)


@app.post("/api/ask")
def ask(body: AskRequest, services: ServicesDep, principal: PrincipalDep) -> Answer:
    try:
        return _pipeline(services, body).answer(body.question, principal=principal, k=body.k)
    except GenerationError as exc:
        # 502: the backend failed, the request was fine. Distinguishable in the
        # UI from "no evidence", which is a successful answer with no content.
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def _attack_corpus() -> dict[str, tuple[Attack, str]]:
    """Every attack that can be run, by id, with the corpus it came from."""
    found: dict[str, tuple[Attack, str]] = {}
    for corpus, directory in ATTACK_DIRS.items():
        if not directory.exists():
            continue
        for attack in load_attacks(directory):
            found[attack.id] = (attack, corpus)
    return found


@app.get("/api/attacks")
def attacks() -> list[AttackView]:
    """The attack corpus, as data rather than as payloads.

    The poison text is deliberately not returned. It is in the repository for
    anyone who wants it; putting it in a JSON response makes this endpoint a
    convenient way to fetch working injection strings from a running service.
    """
    return [
        AttackView(
            id=attack.id,
            family=str(attack.family),
            description=attack.description,
            corpus=corpus,
            target_query=attack.target_query,
            tlp=str(attack.doc.tlp),
            trust_tier=int(attack.doc.trust_tier),
        )
        for attack, corpus in _attack_corpus().values()
    ]


@app.post("/api/attacks/run")
def run_attack(body: AttackRunRequest, services: ServicesDep) -> AttackRunResponse:
    """Run one committed attack against a chosen defence set, for real.

    This indexes the attack's poison document into the live collection, asks
    its query, evaluates the answer and deletes the document again. The
    deletion is in the runner's `finally`, so the poison does not outlive its
    own run even if generation fails.

    Serialised behind a lock: two overlapping runs would each retrieve the
    other's poison. The sink binds a fixed port for the same reason the
    benchmark fixes it -- the poison text embeds that URL, so a changing port
    changes the document and its ranking.
    """
    entry = _attack_corpus().get(body.attack_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown attack {body.attack_id!r}")
    attack, _ = entry

    requested = services.config.defenses if body.defenses is None else body.defenses
    config = config_for(body.retrieval, requested)
    if not ATTACK_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="An attack run is already in progress.")
    try:
        with ExfiltrationSink() as sink:
            runner = factory.build_attack_runner(config, sink=sink)
            result = runner.run(attack)
    except GenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        ATTACK_LOCK.release()
    return AttackRunResponse(
        result=result,
        defenses_applied=list(config.defenses),
        retrieval=body.retrieval or "dense",
    )


@app.get("/api/findings")
def findings(services: ServicesDep) -> findings_module.Findings:
    """What the experiments measured, read back from the committed dumps.

    Deliberately touches no service. The console needs Qdrant and Ollama up to
    show anything; this page has to work when they are down, so the one part
    of the project someone is likely to be shown cannot be taken out by a
    docker daemon that did not start.
    """
    return findings_module.load(services.config.paths.reports_dir / "data")


@app.get("/findings")
def findings_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "findings.html")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
