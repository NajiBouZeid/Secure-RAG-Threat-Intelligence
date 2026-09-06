"""Choosing which stored vectors to attempt to invert.

Inverting a vector is independent of every other vector, so a reconstruction
rate measured on a fair sample is the rate for the whole index. That is what
makes it legitimate to attack a few hundred chunks instead of embedding all
40815 with a second encoder -- the extra vectors would cost an hour of GPU-less
compute to reproduce the same percentage.

Two rules keep the sample fair. The confidential corpus is taken whole, because
34 internal notes are the only chunks whose leakage actually matters and a
proportional draw would take one or two of them; everything else is sampled in
proportion to how much of the index it occupies, so the headline rate is not
skewed by over-weighting whichever corpus inverts best.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence

from pydantic import BaseModel, Field

from threatrag.domain.models import Chunk, SourceType
from threatrag.domain.ports import VectorStore

# Fixed so a re-run selects the same chunks. Changing it changes the sample and
# therefore the numbers, so record it alongside any result.
DEFAULT_SEED = 20260906
DEFAULT_SIZE = 300

# Taken in full rather than sampled: these carry the AMBER and RED material.
CENSUS_SOURCE_TYPES: tuple[SourceType, ...] = (SourceType.INTERNAL_NOTE,)

# M4's poison is cleaned up by its own runner, but a hard-killed run can leave
# it behind, and attacker-authored text in an inversion sample would inflate
# the reconstruction rate with content the attacker already wrote.
EXCLUDED_SOURCE_TYPES: tuple[SourceType, ...] = (SourceType.SYNTHETIC_ADVERSARIAL,)


class SampleReport(BaseModel):
    """The sample plus the census it was drawn from, so the method is auditable."""

    seed: int
    requested: int
    population: dict[str, int] = Field(default_factory=dict)
    selected: dict[str, int] = Field(default_factory=dict)
    chunks: list[Chunk] = Field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.chunks)


def _allocate(counts: dict[str, int], total: int) -> dict[str, int]:
    """Split ``total`` across corpora in proportion to their size.

    Largest-remainder rather than rounding each share independently, so the
    allocated seats sum to exactly ``total`` instead of drifting by a few.
    """
    population = sum(counts.values())
    if population == 0 or total <= 0:
        return {name: 0 for name in counts}

    exact = {name: total * count / population for name, count in counts.items()}
    floors = {name: int(value) for name, value in exact.items()}
    # Never allocate more seats than a corpus has chunks.
    floors = {name: min(value, counts[name]) for name, value in floors.items()}

    remaining = total - sum(floors.values())
    by_remainder = sorted(
        counts, key=lambda name: (exact[name] - int(exact[name]), counts[name]), reverse=True
    )
    for name in by_remainder:
        if remaining <= 0:
            break
        if floors[name] < counts[name]:
            floors[name] += 1
            remaining -= 1
    return floors


def sample_chunks(
    store: VectorStore,
    *,
    size: int = DEFAULT_SIZE,
    seed: int = DEFAULT_SEED,
    census: Sequence[SourceType] = CENSUS_SOURCE_TYPES,
    excluded: Sequence[SourceType] = EXCLUDED_SOURCE_TYPES,
) -> SampleReport:
    """Draw a reproducible, corpus-stratified sample of indexed chunks.

    One pass over the index. Non-census corpora are reservoir-sampled into
    buckets capped at ``size``, which bounds memory at a few hundred chunks
    regardless of index size while still yielding a uniform draw, and gives
    exact populations for the proportional allocation afterwards.
    """
    rng = random.Random(seed)
    census_names = {source.value for source in census}
    excluded_names = {source.value for source in excluded}

    counts: dict[str, int] = {}
    reservoirs: dict[str, list[Chunk]] = {}
    taken_whole: list[Chunk] = []

    for chunk in store.scroll_chunks():
        name = chunk.source_type.value
        if name in excluded_names:
            continue
        counts[name] = counts.get(name, 0) + 1

        if name in census_names:
            taken_whole.append(chunk)
            continue

        bucket = reservoirs.setdefault(name, [])
        if len(bucket) < size:
            bucket.append(chunk)
        else:
            # Algorithm R: the nth item replaces a uniformly chosen slot with
            # probability size/n, which leaves every seen chunk equally likely.
            slot = rng.randrange(counts[name])
            if slot < size:
                bucket[slot] = chunk

    sampled_counts = {name: counts[name] for name in reservoirs}
    quotas = _allocate(sampled_counts, size)

    chosen: list[Chunk] = list(taken_whole)
    for name, bucket in reservoirs.items():
        chosen.extend(rng.sample(bucket, min(quotas[name], len(bucket))))

    chosen.sort(key=lambda chunk: chunk.id)
    selected: dict[str, int] = {}
    for chunk in chosen:
        key = chunk.source_type.value
        selected[key] = selected.get(key, 0) + 1

    return SampleReport(
        seed=seed,
        requested=size,
        population=dict(sorted(counts.items())),
        selected=dict(sorted(selected.items())),
        chunks=chosen,
    )


def secret_terms(chunk: Chunk) -> list[str]:
    """Terms whose appearance in a reconstruction counts as a real disclosure.

    Reconstruction quality and disclosure are different questions: a paraphrase
    scoring well on BLEU may leak nothing, while a mangled sentence that still
    names the actor and the CVE leaks everything that mattered.

    Two rules keep the number meaningful, both learned by getting it wrong.

    Declared terms are scoped to the chunk that actually contains them. They are
    carried on the *document*, so without this filter a chunk would be scored on
    secrets it never held -- chunk 3 of an incident report marked down for
    failing to disclose a hostname that only ever appeared in chunk 1.

    The source reference counts only when the text carries it. For a public
    corpus that identifier is the subject and recovering it is a real signal,
    but for an internal note it is a filing number: scoring it as the secret is
    what made M5's first inversion run report zero leakage from reconstructions
    that plainly leaked the technique and the hostname.
    """
    declared: list[str] = [chunk.source_ref]
    metadata: Iterable[tuple[str, str | list[str]]] = chunk.metadata.items()
    for key, value in metadata:
        if key in {"secret_terms", "indicators", "actors"}:
            declared.extend([value] if isinstance(value, str) else value)

    haystack = chunk.text.lower()
    return [term for term in dict.fromkeys(declared) if term and term.lower() in haystack]
