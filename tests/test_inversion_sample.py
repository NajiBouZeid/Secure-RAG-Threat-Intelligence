"""Inversion target sampling.

The sample is the whole basis for claiming a reconstruction rate measured on a
few hundred chunks describes an index of forty thousand, so what is under test
is fairness and reproducibility: the confidential corpus is never diluted away,
the rest is drawn in proportion, and the same seed picks the same chunks.
"""

from __future__ import annotations

from collections.abc import Iterator

from threatrag.domain.models import TLP, Chunk, SourceType
from threatrag.security.inversion.sample import _allocate, sample_chunks, secret_terms


def _chunk(index: int, source_type: SourceType) -> Chunk:
    return Chunk(
        id=f"{source_type.value}-{index:05d}#0",
        doc_id=f"{source_type.value}-{index:05d}",
        ordinal=0,
        text=f"body {index}",
        title=f"title {index}",
        source_type=source_type,
        source_ref=f"REF-{index}",
        tlp=TLP.CLEAR,
    )


class FakeStore:
    """Only scroll_chunks is exercised; the rest satisfies the protocol shape."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

    def scroll_chunks(
        self, *, source_type: str | None = None, batch_size: int = 256
    ) -> Iterator[Chunk]:
        for chunk in self.chunks:
            if source_type is None or chunk.source_type.value == source_type:
                yield chunk


def _population() -> list[Chunk]:
    chunks: list[Chunk] = []
    chunks += [_chunk(i, SourceType.ATTACK_CTI) for i in range(2000)]
    chunks += [_chunk(i, SourceType.NVD_CVE) for i in range(1000)]
    chunks += [_chunk(i, SourceType.VENDOR_REPORT) for i in range(200)]
    chunks += [_chunk(i, SourceType.INTERNAL_NOTE) for i in range(34)]
    return chunks


def test_the_confidential_corpus_is_taken_whole() -> None:
    """A proportional draw would take one note out of 34; those are the ones that matter."""
    report = sample_chunks(FakeStore(_population()), size=100)

    assert report.selected[SourceType.INTERNAL_NOTE.value] == 34


def test_other_corpora_are_drawn_in_proportion() -> None:
    report = sample_chunks(FakeStore(_population()), size=320)

    # 2000 : 1000 : 200 of 3200 sampled chunks, for 320 seats.
    assert report.selected[SourceType.ATTACK_CTI.value] == 200
    assert report.selected[SourceType.NVD_CVE.value] == 100
    assert report.selected[SourceType.VENDOR_REPORT.value] == 20


def test_the_sample_size_is_the_quota_plus_the_census() -> None:
    report = sample_chunks(FakeStore(_population()), size=300)

    assert report.size == 300 + 34


def test_the_same_seed_picks_the_same_chunks() -> None:
    first = sample_chunks(FakeStore(_population()), size=100, seed=7)
    second = sample_chunks(FakeStore(_population()), size=100, seed=7)

    assert [chunk.id for chunk in first.chunks] == [chunk.id for chunk in second.chunks]


def test_a_different_seed_picks_a_different_sample() -> None:
    first = sample_chunks(FakeStore(_population()), size=100, seed=7)
    second = sample_chunks(FakeStore(_population()), size=100, seed=8)

    assert [chunk.id for chunk in first.chunks] != [chunk.id for chunk in second.chunks]


def test_attacker_authored_documents_are_excluded() -> None:
    """Leftover M4 poison would inflate the rate with text the attacker wrote."""
    population = _population() + [_chunk(i, SourceType.SYNTHETIC_ADVERSARIAL) for i in range(50)]

    report = sample_chunks(FakeStore(population), size=100)

    assert SourceType.SYNTHETIC_ADVERSARIAL.value not in report.selected
    assert SourceType.SYNTHETIC_ADVERSARIAL.value not in report.population


def test_the_report_records_the_population_it_drew_from() -> None:
    report = sample_chunks(FakeStore(_population()), size=100)

    assert report.population[SourceType.ATTACK_CTI.value] == 2000
    assert report.population[SourceType.INTERNAL_NOTE.value] == 34
    assert report.seed != 0


def test_allocation_hands_out_exactly_the_seats_available() -> None:
    quotas = _allocate({"a": 7, "b": 5, "c": 3}, 10)

    assert sum(quotas.values()) == 10


def test_allocation_never_exceeds_a_corpus_that_is_too_small() -> None:
    quotas = _allocate({"a": 1000, "b": 2}, 100)

    assert quotas["b"] <= 2
    assert sum(quotas.values()) == 100


def test_secret_terms_include_the_reference_and_declared_indicators() -> None:
    chunk = _chunk(1, SourceType.INTERNAL_NOTE).model_copy(
        update={"metadata": {"secret_terms": ["41 repositories", "beacon.exe"]}}
    )

    assert secret_terms(chunk) == ["REF-1", "41 repositories", "beacon.exe"]
