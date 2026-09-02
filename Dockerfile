FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl \
 && rm -rf /var/lib/apt/lists/*

# Dependency layer first so source edits don't reinstall the world.
COPY pyproject.toml README.md ./
COPY src/threatrag/__init__.py src/threatrag/__init__.py
RUN pip install --upgrade pip && pip install -e ".[pdf]"

COPY . .
RUN pip install -e ".[pdf]"

EXPOSE 8000
CMD ["uvicorn", "threatrag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
