"""Packaging the attacker's view of the index for an off-machine inversion run.

Two bundles, deliberately separated. ``vectors.npy`` and ``vector_ids.json``
are the attack input and hold nothing but numbers and opaque ids: that is
exactly what someone who stole the collection's vectors would have, and it is
what gets uploaded to whatever GPU runs vec2text. ``truth.jsonl`` never leaves
this machine -- it is the answer key, and scoring against it happens locally.

Handing the reconstruction step the source text would make the result circular,
so the split is a property of the experiment rather than tidiness.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from pydantic import BaseModel, Field

from threatrag.domain.ports import VectorStore
from threatrag.domain.types import Matrix, Vector
from threatrag.security.inversion.sample import SampleReport, secret_terms


class TruncatingEmbedder(Protocol):
    """An embedder that can say what text it actually saw after truncation."""

    @property
    def name(self) -> str: ...

    @property
    def max_tokens(self) -> int | None: ...

    def embed_documents(self, texts: Sequence[str]) -> Matrix: ...

    def tokenized_prefix(self, text: str) -> str: ...


VECTORS_FILE = "vectors.npy"
IDS_FILE = "vector_ids.json"
TRUTH_FILE = "truth.jsonl"
MANIFEST_FILE = "manifest.json"


class ExportManifest(BaseModel):
    """What was exported, and what was asked for but not there."""

    vector_name: str
    dim: int
    exported: int
    # Which of the three bundles this is. They differ in ways that matter
    # separately, so a result is meaningless without knowing which it scored.
    variant: str = "stored"
    missing: list[str] = Field(default_factory=list)
    seed: int = 0
    # Set on the control bundle only: the token budget its vectors were
    # embedded under, which is also the length its answer key was cut to.
    max_tokens: int | None = None
    population: dict[str, int] = Field(default_factory=dict)
    selected: dict[str, int] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def upload_files(self) -> tuple[str, str]:
        """The only two files that may leave the machine."""
        return (VECTORS_FILE, IDS_FILE)


def export_control(
    sample: SampleReport,
    embedder: TruncatingEmbedder,
    *,
    out_dir: Path,
    variant: str,
) -> ExportManifest:
    """A second bundle embedded at the corrector's own training length.

    The public GTR corrector was fitted on 32-token sequences and these chunks
    are 512 characters, so a weak result on the main bundle has two possible
    causes that matter very differently: the attack does not work, or the text
    is longer than the attack was built for. This bundle separates them by
    running the same attack on the length it was designed for, and is the upper
    bound the main result should be read against.

    The answer key holds the *decoded truncated prefix*, not the full chunk.
    Scoring a 32-token vector against 512 characters would mark the
    reconstruction wrong for omitting words its vector never carried.

    The vectors are embedded from the **original** text, letting the tokenizer
    truncate, and the decoded prefix is used only as the answer key. Embedding
    the prefix instead would push the text through a decode-and-re-encode round
    trip, which sentencepiece does not round-trip exactly: it renormalises
    whitespace, so the bundle would encode subtly different text from the one
    the index holds. Measured on a real chunk, that shifted the vector's norm
    from 0.3731 to 0.386 and cost 5% of the cosine against the stored vector --
    enough to contaminate the very comparison these bundles exist to make.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = list(sample.chunks)
    prefixes = [embedder.tokenized_prefix(chunk.text) for chunk in chunks]
    matrix = (
        embedder.embed_documents([chunk.text for chunk in chunks]).astype(np.float32)
        if chunks
        else np.empty((0, 0), dtype=np.float32)
    )

    np.save(out_dir / VECTORS_FILE, matrix)
    _write_json(out_dir / IDS_FILE, [chunk.id for chunk in chunks])

    with (out_dir / TRUTH_FILE).open("w", encoding="utf-8", newline="\n") as handle:
        for chunk, prefix in zip(chunks, prefixes, strict=True):
            record = {
                "chunk_id": chunk.id,
                "doc_id": chunk.doc_id,
                "source_type": chunk.source_type.value,
                "source_ref": chunk.source_ref,
                "title": chunk.title,
                "tlp": chunk.tlp.value,
                "text": prefix,
                "secret_terms": [
                    term for term in secret_terms(chunk) if term.lower() in prefix.lower()
                ],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = ExportManifest(
        vector_name=embedder.name,
        dim=int(matrix.shape[1]) if matrix.size else 0,
        exported=len(chunks),
        variant=variant,
        seed=sample.seed,
        population=sample.population,
        selected=sample.selected,
        max_tokens=embedder.max_tokens,
    )
    _write_json(out_dir / MANIFEST_FILE, manifest.model_dump(mode="json"))
    return manifest


def export_targets(
    store: VectorStore,
    sample: SampleReport,
    *,
    vector_name: str,
    out_dir: Path,
) -> ExportManifest:
    """Read the sampled vectors back out of the index and write the bundles.

    Vectors are read from the store rather than recomputed, so what is attacked
    is the value actually sitting in the database and not a fresh embedding that
    happens to agree with it. That distinction turned out to matter: a Qdrant
    collection using cosine distance **normalises vectors on write**, verified
    against a live instance by storing a norm-5 vector and reading back a norm-1
    one. Whoever steals this index gets directions, not magnitudes, and the
    corrector was trained on unnormalised embeddings.

    So this bundle is the honest realistic case, and it is also not enough on
    its own -- see ``export_control`` for the two bundles that let a failure
    here be attributed to length or to that lost magnitude rather than to the
    attack.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_ids = [chunk.id for chunk in sample.chunks]
    found = store.get_vectors(vector_name, chunk_ids)

    ordered = [chunk for chunk in sample.chunks if chunk.id in found]
    missing = [chunk_id for chunk_id in chunk_ids if chunk_id not in found]

    dim = int(next(iter(found.values())).shape[0]) if found else 0
    matrix = (
        np.stack([found[chunk.id] for chunk in ordered]).astype(np.float32)
        if ordered
        else np.empty((0, dim), dtype=np.float32)
    )
    np.save(out_dir / VECTORS_FILE, matrix)

    _write_json(out_dir / IDS_FILE, [chunk.id for chunk in ordered])

    with (out_dir / TRUTH_FILE).open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in ordered:
            record = {
                "chunk_id": chunk.id,
                "doc_id": chunk.doc_id,
                "source_type": chunk.source_type.value,
                "source_ref": chunk.source_ref,
                "title": chunk.title,
                "tlp": chunk.tlp.value,
                "text": chunk.text,
                "secret_terms": secret_terms(chunk),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = ExportManifest(
        vector_name=vector_name,
        dim=dim,
        exported=len(ordered),
        missing=missing,
        seed=sample.seed,
        population=sample.population,
        selected=sample.selected,
    )
    _write_json(out_dir / MANIFEST_FILE, manifest.model_dump(mode="json"))
    return manifest


def load_truth(path: Path) -> dict[str, dict[str, Any]]:
    """Read the answer key back, keyed by chunk id."""
    truth: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            truth[str(record["chunk_id"])] = record
    return truth


def load_vectors(out_dir: Path) -> dict[str, Vector]:
    """Read the exported attack bundle back, keyed by chunk id."""
    ids: list[str] = json.loads((out_dir / IDS_FILE).read_text(encoding="utf-8"))
    matrix = np.load(out_dir / VECTORS_FILE)
    return {chunk_id: matrix[index].astype(np.float32) for index, chunk_id in enumerate(ids)}


def load_reconstructions(path: Path) -> dict[str, str]:
    """Read what came back from the inversion run.

    One JSON object per line with ``chunk_id`` and ``reconstruction``, which is
    what the GPU-side script writes; a run that produced nothing for a target
    simply omits it rather than emitting an empty string, so a failed
    reconstruction is never scored as an empty one.
    """
    found: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("reconstruction")
            if text is None:
                continue
            found[str(record["chunk_id"])] = str(text)
    return found


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
