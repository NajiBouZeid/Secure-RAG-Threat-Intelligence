"""Re-identifying stolen vectors without inverting them.

No public vec2text corrector exists for all-MiniLM-L6-v2, which is the model
this system actually retrieves with, so the reconstruction attack does not run
against the live index at all. That is a real limit on the attacker and it is
reported as one -- but it is not the same as the index being safe, and stopping
at "inversion is impossible" would overstate what the negative result shows.

There is a cheaper attack that needs no corrector and works against any
encoder. Most of this corpus is public: MITRE publishes ATT&CK, NVD publishes
CVEs, and all-MiniLM-L6-v2 is a public checkpoint. An attacker holding stolen
vectors can embed the public sources himself and match each stolen vector to
its nearest public document. Nothing is reconstructed; the vector is simply
recognised.

The confidential notes are the interesting half, because nothing in the public
corpus matches them. They cannot be recognised -- but their nearest public
neighbour still names the technique the note is about, so the attacker learns
the subject of a document he cannot read. That is disclosure without inversion,
and it is measured separately here.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, Field

from threatrag.domain.models import Chunk, Document
from threatrag.domain.ports import Embedder
from threatrag.domain.types import Matrix, Vector
from threatrag.security.inversion.score import tokenize

# Titles arrive as "T1003.001 LSASS Memory (technique)" and
# "CVE-2026-26125 (CRITICAL)". The identifier and the parenthesised kind are
# scaffolding this project added at ingest, not subject matter: matching on
# them would score "technique" or "critical" appearing anywhere in a note as
# topic disclosure.
_PARENTHESISED = re.compile(r"\([^)]*\)")

# Short tokens and bare numbers say nothing about subject matter, and a note
# dated 2026-02-11 would otherwise match every CVE published in 2026.
_MIN_TERM_LENGTH = 4
_STOPWORDS = frozenset(
    {"with", "from", "that", "this", "into", "over", "their", "them", "using", "used", "have"}
)


def topic_terms(title: str, source_ref: str) -> list[str]:
    """Content words from a document's name, with the identifier stripped.

    A CVE title reduces to nothing once its id and severity are removed, which
    is the conservative answer: the id alone names the subject only to someone
    who then looks it up.
    """
    text = title
    if source_ref and text.lower().startswith(source_ref.lower()):
        text = text[len(source_ref) :]
    text = _PARENTHESISED.sub(" ", text)
    return [
        token
        for token in tokenize(text)
        if len(token) >= _MIN_TERM_LENGTH
        and token not in _STOPWORDS
        and any(character.isalpha() for character in token)
    ]


class ReferenceCorpus(BaseModel):
    """What the attacker built for himself out of public sources."""

    doc_ids: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    titles: list[str] = Field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.doc_ids)


class ReidentifyRow(BaseModel):
    chunk_id: str
    doc_id: str
    source_type: str
    tlp: str
    predicted_doc_id: str
    predicted_ref: str
    predicted_title: str = ""
    score: float
    in_reference: bool
    correct: bool = False
    # The neighbour's name shares a content word with the chunk: the attacker
    # has the subject of a document he cannot read.
    topic_hit: bool = False
    # The stricter question -- the chunk quotes the neighbour's identifier
    # outright. Reported alongside because analysts write prose, so this
    # undercounts disclosure badly and it is worth showing by how much.
    ref_hit: bool = False


class PopulationScore(BaseModel):
    name: str
    count: int = 0
    recognised: int = 0
    topic_hits: int = 0
    ref_hits: int = 0


class ReidentifyReport(BaseModel):
    rows: list[ReidentifyRow] = Field(default_factory=list)
    reference_size: int = 0
    # Over chunks whose document is in the public corpus: was it recognised?
    recognised: int = 0
    recognisable: int = 0
    # Over chunks whose document is not: did the nearest public neighbour still
    # name what the document is about?
    topic_hits: int = 0
    ref_hits: int = 0
    unrecognisable: int = 0
    by_source_type: list[PopulationScore] = Field(default_factory=list)

    @property
    def top1_accuracy(self) -> float:
        return self.recognised / self.recognisable if self.recognisable else 0.0

    @property
    def topic_disclosure_rate(self) -> float:
        return self.topic_hits / self.unrecognisable if self.unrecognisable else 0.0


def build_reference(
    documents: Iterable[Document],
    embedder: Embedder,
    *,
    batch_size: int = 128,
) -> tuple[ReferenceCorpus, Matrix]:
    """Embed public source documents as the attacker would.

    Whole documents, not this system's chunks. The attacker has the published
    sources, not the ingest configuration, and matching stolen chunk vectors
    against the very chunks they came from would measure nothing but the
    identity function.
    """
    doc_ids: list[str] = []
    source_refs: list[str] = []
    titles: list[str] = []
    texts: list[str] = []
    blocks: list[Matrix] = []

    def flush() -> None:
        if texts:
            blocks.append(embedder.embed_documents(texts))
            texts.clear()

    for document in documents:
        doc_ids.append(document.id)
        source_refs.append(document.source_ref)
        titles.append(document.title)
        texts.append(document.text)
        if len(texts) >= batch_size:
            flush()
    flush()

    matrix = (
        np.vstack(blocks).astype(np.float32)
        if blocks
        else np.empty((0, embedder.dim), dtype=np.float32)
    )
    corpus = ReferenceCorpus(doc_ids=doc_ids, source_refs=source_refs, titles=titles)
    return corpus, _normalize(matrix)


def reidentify(
    targets: Sequence[Chunk],
    stolen: Mapping[str, Vector],
    corpus: ReferenceCorpus,
    reference: Matrix,
) -> ReidentifyReport:
    """Match each stolen vector to its nearest public document.

    A chunk counts as recognised when the nearest public document is the one it
    was actually chunked from. A chunk whose document is not public cannot be
    recognised, so it is scored on the weaker question instead: does the
    neighbour's name share a content word with the chunk, meaning the attacker
    has learned the subject of a document he cannot read.
    """
    report = ReidentifyReport(reference_size=corpus.size)
    if corpus.size == 0:
        return report

    public_docs = set(corpus.doc_ids)
    scorable = [chunk for chunk in targets if chunk.id in stolen]
    if not scorable:
        return report

    queries = _normalize(np.stack([stolen[chunk.id] for chunk in scorable]).astype(np.float32))
    # Vectors are unit-normalised, so a dot product is the cosine.
    similarities = queries @ reference.T
    best = np.argmax(similarities, axis=1)

    rows: list[ReidentifyRow] = []
    for position, chunk in enumerate(scorable):
        index = int(best[position])
        predicted_ref = corpus.source_refs[index]
        predicted_title = corpus.titles[index] if corpus.titles else ""
        in_reference = chunk.doc_id in public_docs
        haystack = set(tokenize(chunk.text))

        rows.append(
            ReidentifyRow(
                chunk_id=chunk.id,
                doc_id=chunk.doc_id,
                source_type=chunk.source_type.value,
                tlp=chunk.tlp.value,
                predicted_doc_id=corpus.doc_ids[index],
                predicted_ref=predicted_ref,
                predicted_title=predicted_title,
                score=float(similarities[position, index]),
                in_reference=in_reference,
                correct=in_reference and corpus.doc_ids[index] == chunk.doc_id,
                topic_hit=(
                    not in_reference
                    and any(
                        term in haystack for term in topic_terms(predicted_title, predicted_ref)
                    )
                ),
                ref_hit=(
                    not in_reference
                    and bool(predicted_ref)
                    and predicted_ref.lower() in chunk.text.lower()
                ),
            )
        )

    hidden = [row for row in rows if not row.in_reference]
    return report.model_copy(
        update={
            "rows": rows,
            "recognisable": sum(1 for row in rows if row.in_reference),
            "recognised": sum(1 for row in rows if row.correct),
            "unrecognisable": len(hidden),
            "topic_hits": sum(1 for row in hidden if row.topic_hit),
            "ref_hits": sum(1 for row in hidden if row.ref_hit),
            "by_source_type": [
                PopulationScore(
                    name=name,
                    count=sum(1 for row in rows if row.source_type == name),
                    recognised=sum(1 for row in rows if row.source_type == name and row.correct),
                    topic_hits=sum(
                        1 for row in hidden if row.source_type == name and row.topic_hit
                    ),
                    ref_hits=sum(1 for row in hidden if row.source_type == name and row.ref_hit),
                )
                for name in sorted({row.source_type for row in rows})
            ],
        }
    )


def _normalize(matrix: Matrix) -> Matrix:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # A zero vector would divide to nan and then win every argmax.
    norms[norms == 0] = 1.0
    normalized: Matrix = (matrix / norms).astype(np.float32)
    return normalized
