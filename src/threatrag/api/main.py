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

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from threatrag import __version__, factory
from threatrag.config import Config, load_config
from threatrag.domain.models import Answer, Principal, RetrievedChunk
from threatrag.rag.generators.ollama import GenerationError
from threatrag.rag.pipeline import AnswerPipeline

STATIC_DIR = Path(__file__).parent / "static"

PRINCIPAL_HEADER = "X-Principal-Id"


@dataclass(slots=True)
class Services:
    config: Config
    pipeline: AnswerPipeline


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Built once at startup: the pipeline holds an embedder whose model would
    # otherwise be loaded from disk on every request.
    config = load_config()
    app.state.services = Services(config=config, pipeline=factory.build_answer_pipeline(config))
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


class PrincipalView(BaseModel):
    id: str
    label: str
    role: str
    clearance: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/api/principals")
def principals(services: ServicesDep) -> list[PrincipalView]:
    """The identities the UI offers. Labels come from config, not the client."""
    return [
        PrincipalView(id=key, label=spec.label, role=spec.role, clearance=spec.clearance.value)
        for key, spec in services.config.principals.items()
    ]


@app.post("/api/search")
def search(
    body: AskRequest, services: ServicesDep, principal: PrincipalDep
) -> list[RetrievedChunk]:
    """Retrieval only. Useful for seeing what a clearance can reach without
    paying for generation, and for telling a retrieval failure apart from a
    generation one."""
    return services.pipeline.retrieve(body.question, principal=principal, k=body.k)


@app.post("/api/ask")
def ask(body: AskRequest, services: ServicesDep, principal: PrincipalDep) -> Answer:
    try:
        return services.pipeline.answer(body.question, principal=principal, k=body.k)
    except GenerationError as exc:
        # 502: the backend failed, the request was fine. Distinguishable in the
        # UI from "no evidence", which is a successful answer with no content.
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
