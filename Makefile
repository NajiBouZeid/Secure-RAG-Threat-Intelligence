.DEFAULT_GOAL := help
PY ?= python

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package with dev + pdf extras
	$(PY) -m pip install -e ".[dev,pdf]"

up:  ## Start Qdrant (and the API) via docker compose
	docker compose up -d qdrant

down:  ## Stop the stack
	docker compose down

fetch:  ## Download raw corpora into data/raw
	$(PY) -m threatrag.cli fetch attack

ingest:  ## Parse, chunk, embed and index the corpora
	$(PY) -m threatrag.cli ingest

query:  ## One-off retrieval check: make query Q="APT29 persistence"
	$(PY) -m threatrag.cli query "$(Q)"

eval-retrieval:  ## Recall@k / MRR / nDCG on the ATT&CK gold set
	$(PY) -m threatrag.cli eval-retrieval

serve:  ## Run the API locally
	uvicorn threatrag.api.main:app --reload --port 8000

test:  ## Run the test suite
	pytest

lint:  ## Ruff + mypy
	ruff check src tests && ruff format --check src tests && mypy src

fmt:  ## Autoformat
	ruff format src tests && ruff check --fix src tests

.PHONY: help install up down fetch ingest query eval-retrieval serve test lint fmt
