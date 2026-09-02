"""FastAPI surface. Fleshed out in M2; the health route exists so docker compose
has something real to wait on."""

from __future__ import annotations

from fastapi import FastAPI

from threatrag import __version__

app = FastAPI(title="Secure Threat Intelligence RAG", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
