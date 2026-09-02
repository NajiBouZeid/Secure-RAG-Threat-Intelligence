"""Chunking strategies.

Chunking is a benchmark axis, not a constant: chunk too large and retrieval
returns mostly irrelevant text, chunk too small and the generator loses the
context that makes an answer correct. Every strategy here implements the same
:class:`~threatrag.domain.ports.Chunker` protocol so the retrieval evaluation
can sweep across them without touching the pipeline.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable

from threatrag.domain.models import Chunk, Document
from threatrag.domain.ports import Chunker

# Split points in descending order of how much structure they preserve.
_RECURSIVE_SEPARATORS: tuple[str, ...] = ("\n## ", "\n\n", "\n", ". ", " ")
_SECTION_PATTERN = re.compile(r"^## ", re.MULTILINE)


def _chunk_id(doc_id: str, ordinal: int, text: str) -> str:
    """Content-addressed id: re-ingesting unchanged text reuses the same point.

    Without the digest, an edited document would leave stale chunks behind at
    the same ids -- which in this project is not just untidy but a way for
    poisoned content to survive a re-ingest.
    """
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{doc_id}#{ordinal}:{digest}"


def _build(document: Document, pieces: list[str]) -> list[Chunk]:
    return [
        Chunk(
            id=_chunk_id(document.id, ordinal, text),
            doc_id=document.id,
            ordinal=ordinal,
            text=text,
            title=document.title,
            source_type=document.source_type,
            source_ref=document.source_ref,
            url=document.url,
            tlp=document.tlp,
            trust_tier=document.trust_tier,
            metadata=document.metadata,
        )
        for ordinal, text in enumerate(piece.strip() for piece in pieces)
        if text
    ]


def _pack(units: list[str], joiner: str, chunk_size: int, overlap: int) -> list[str]:
    """Greedily merge ``units`` into pieces of at most ``chunk_size`` characters.

    Overlap is carried as whole trailing units rather than a raw character
    slice, so a chunk never begins mid-word.
    """
    pieces: list[str] = []
    current: list[str] = []
    length = 0

    for unit in units:
        unit_len = len(unit) + len(joiner)
        if current and length + unit_len > chunk_size:
            pieces.append(joiner.join(current))
            carried: list[str] = []
            carried_len = 0
            for previous in reversed(current):
                if carried_len + len(previous) > overlap:
                    break
                carried.insert(0, previous)
                carried_len += len(previous) + len(joiner)
            current = carried
            length = carried_len
        current.append(unit)
        length += unit_len

    if current:
        pieces.append(joiner.join(current))
    return pieces


class StructuralChunker:
    """Split on the ``## `` section headings the sources emit.

    ATT&CK entries are already short and semantically partitioned
    (Description / Detection / Tactics), so honouring that structure beats any
    character-count heuristic. Oversized sections fall through to recursive
    splitting.
    """

    name = "structural"

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self._chunk_size = chunk_size
        self._overlap = chunk_overlap
        self._fallback = RecursiveChunker(chunk_size, chunk_overlap)

    def split(self, document: Document) -> list[Chunk]:
        text = document.text
        matches = list(_SECTION_PATTERN.finditer(text))
        if not matches:
            return self._fallback.split(document)

        header = text[: matches[0].start()].strip()
        bounds = [match.start() for match in matches] + [len(text)]
        sections = [text[bounds[i] : bounds[i + 1]].strip() for i in range(len(matches))]

        pieces: list[str] = []
        for sec in sections:
            # The document title is prepended to every chunk: an isolated
            # "Detection" block is close to unretrievable without it.
            scoped = f"{header}\n\n{sec}" if header else sec
            if len(scoped) <= self._chunk_size:
                pieces.append(scoped)
            else:
                pieces.extend(
                    f"{header}\n\n{part}" if header else part
                    for part in _split_recursive(sec, self._chunk_size, self._overlap)
                )
        return _build(document, pieces)


class RecursiveChunker:
    """Split on progressively finer separators until pieces fit."""

    name = "recursive"

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self._chunk_size = chunk_size
        self._overlap = chunk_overlap

    def split(self, document: Document) -> list[Chunk]:
        return _build(document, _split_recursive(document.text, self._chunk_size, self._overlap))


class FixedChunker:
    """Fixed-width character windows. The naive baseline the others are scored against."""

    name = "fixed"

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self._chunk_size = chunk_size
        self._overlap = min(chunk_overlap, chunk_size - 1)

    def split(self, document: Document) -> list[Chunk]:
        text = document.text
        stride = self._chunk_size - self._overlap
        pieces = [text[i : i + self._chunk_size] for i in range(0, max(len(text), 1), stride)]
        return _build(document, pieces)


def _split_recursive(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    for separator in _RECURSIVE_SEPARATORS:
        units = [unit for unit in text.split(separator) if unit.strip()]
        if len(units) < 2:
            continue
        packed = _pack(units, separator, chunk_size, overlap)
        if all(len(piece) <= chunk_size for piece in packed):
            return packed
        # Separator helped but left oversized pieces: recurse into just those.
        result: list[str] = []
        for piece in packed:
            result.extend(
                [piece]
                if len(piece) <= chunk_size
                else _split_recursive(piece, chunk_size, overlap)
            )
        return result

    # No separator applies (one very long token): fall back to hard slicing.
    return [text[i : i + chunk_size] for i in range(0, len(text), max(chunk_size - overlap, 1))]


CHUNKERS: dict[str, Callable[[int, int], Chunker]] = {
    StructuralChunker.name: StructuralChunker,
    RecursiveChunker.name: RecursiveChunker,
    FixedChunker.name: FixedChunker,
}


def build_chunker(strategy: str, chunk_size: int, chunk_overlap: int) -> Chunker:
    """Instantiate a chunker by name."""
    if strategy not in CHUNKERS:
        raise KeyError(f"Unknown chunking strategy {strategy!r}; have {sorted(CHUNKERS)}")
    return CHUNKERS[strategy](chunk_size, chunk_overlap)
