"""Answer pipeline tests.

The generator is stubbed throughout. What is under test is the wiring the
security work depends on -- that an empty retrieval never reaches the model,
that the principal reaches the store, that citation auditing records rather
than repairs -- none of which needs a live LLM and all of which would be
untestable in CI if it did.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from threatrag.domain.models import (
    TLP,
    Answer,
    Chunk,
    Principal,
    RetrievedChunk,
    SourceType,
)
from threatrag.domain.types import Matrix, Vector
from threatrag.rag.pipeline import NO_EVIDENCE, AnswerPipeline, format_context, resolve_citations
from threatrag.rag.retriever import Retriever


def _chunk(ref: str, text: str = "body", tlp: TLP = TLP.CLEAR) -> Chunk:
    return Chunk(
        id=f"{ref}#0",
        doc_id=ref,
        ordinal=0,
        text=text,
        title=f"{ref} title",
        source_type=SourceType.ATTACK_CTI,
        source_ref=ref,
        tlp=tlp,
    )


def _hits(*refs: str) -> list[RetrievedChunk]:
    return [RetrievedChunk(chunk=_chunk(ref), score=1.0) for ref in refs]


class StubEmbedder:
    name = "stub"
    dim = 3

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        return np.zeros((len(texts), 3), dtype=np.float32)

    def embed_query(self, text: str) -> Vector:
        return np.zeros(3, dtype=np.float32)


class StubStore:
    """Records the principal it was searched with; returns a fixed result set."""

    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results
        self.seen_principal: Principal | None = None

    def ensure_collection(self, vectors: object) -> None: ...

    def upsert(self, vector_name: str, chunks: object, vectors: object) -> int:
        return 0

    def search(
        self,
        vector_name: str,
        query_vector: Vector,
        k: int,
        principal: Principal | None = None,
    ) -> list[RetrievedChunk]:
        self.seen_principal = principal
        return self.results[:k]

    def count(self) -> int:
        return len(self.results)

    def delete_by_source_type(self, source_type: str) -> int:
        return 0


class StubGenerator:
    model = "stub-model"

    def __init__(self, reply: str = "answer [1]") -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.reply


def _pipeline(
    results: list[RetrievedChunk], reply: str = "answer [1]", **kwargs: object
) -> tuple[AnswerPipeline, StubStore, StubGenerator]:
    store = StubStore(results)
    generator = StubGenerator(reply)
    retriever = Retriever(StubEmbedder(), store, top_k=5)  # type: ignore[arg-type]
    pipeline = AnswerPipeline(retriever, generator, **kwargs)  # type: ignore[arg-type]
    return pipeline, store, generator


def test_answer_cites_the_retrieved_passage() -> None:
    pipeline, _, _ = _pipeline(_hits("T1055"), reply="Process injection [1].")
    answer = pipeline.answer("how?")
    assert answer.citations == ["T1055 — T1055 title"]
    assert answer.unsupported_citations == []


def test_empty_retrieval_never_reaches_the_model() -> None:
    """With no context the model would answer from memory -- the core failure."""
    pipeline, _, generator = _pipeline([])
    answer = pipeline.answer("how?")
    assert answer.text == NO_EVIDENCE
    assert answer.retrieved == []
    assert generator.calls == []


def test_principal_reaches_the_store() -> None:
    pipeline, store, _ = _pipeline(_hits("T1055"))
    principal = Principal(id="analyst", clearance=TLP.AMBER)
    pipeline.answer("how?", principal=principal)
    assert store.seen_principal is principal


def test_invented_marker_is_recorded_not_removed() -> None:
    pipeline, _, _ = _pipeline(_hits("T1055"), reply="Claim [1]. Other claim [7].")
    answer = pipeline.answer("how?")
    assert answer.citations == ["T1055 — T1055 title"]
    assert answer.unsupported_citations == ["[7]"]
    # Not enforcement: the text is returned exactly as the model produced it.
    assert "[7]" in answer.text


def test_defense_hook_runs_on_the_answer() -> None:
    class Redactor:
        name = "redactor"

        def on_answer(self, answer: Answer) -> Answer:
            return answer.model_copy(update={"text": "[redacted]", "blocked": True})

    pipeline, _, _ = _pipeline(_hits("T1055"), defenses=[Redactor()])
    answer = pipeline.answer("how?")
    assert answer.text == "[redacted]"
    assert answer.blocked is True
    assert answer.defenses_applied == ["redactor"]


def test_context_is_numbered_from_one() -> None:
    block = format_context(_hits("A", "B"), max_chars=10_000)
    assert block.startswith("[1] A — A title")
    assert "[2] B — B title" in block


def test_context_budget_drops_whole_passages() -> None:
    hits = [RetrievedChunk(chunk=_chunk(ref, "x" * 200), score=1.0) for ref in ("A", "B", "C")]
    block = format_context(hits, max_chars=260)
    assert "[1]" in block
    assert "[2]" not in block
    # A partial passage would still cost context while being uncitable.
    assert block.count("x") == 200


def test_resolve_citations_deduplicates() -> None:
    resolved, unsupported = resolve_citations("a [1] b [1] c [2]", _hits("A", "B"))
    assert resolved == ["A — A title", "B — B title"]
    assert unsupported == []
