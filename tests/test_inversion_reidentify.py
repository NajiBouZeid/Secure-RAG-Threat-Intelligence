"""Re-identification of stolen vectors against a public corpus.

The embedder is a stub with hand-placed vectors, so the geometry is explicit
and the assertions are about the attack's logic rather than about how MiniLM
happens to rank anything.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from threatrag.domain.models import TLP, Chunk, Document, SourceType
from threatrag.domain.types import Matrix, Vector
from threatrag.security.inversion.reidentify import build_reference, reidentify, topic_terms


class PlacedEmbedder:
    """Returns whatever vector was assigned to a text; unknown text lands at the origin."""

    name = "stub"
    dim = 3

    def __init__(self, placements: dict[str, list[float]]) -> None:
        self.placements = placements
        self.batches: list[int] = []

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        self.batches.append(len(texts))
        return np.array(
            [self.placements.get(text, [0.0, 0.0, 0.0]) for text in texts], dtype=np.float32
        )

    def embed_query(self, text: str) -> Vector:
        return self.embed_documents([text])[0]


def _document(doc_id: str, ref: str, text: str, title: str | None = None) -> Document:
    # Titles carry the identifier and a parenthesised kind, exactly as the
    # ATT&CK ingest writes them, because that is what topic terms are cut from.
    return Document(
        id=doc_id,
        title=title if title is not None else f"{ref} (technique)",
        text=text,
        source_type=SourceType.ATTACK_CTI,
        source_ref=ref,
    )


def _chunk(
    chunk_id: str,
    doc_id: str,
    text: str,
    *,
    source_type: SourceType = SourceType.ATTACK_CTI,
    tlp: TLP = TLP.CLEAR,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        doc_id=doc_id,
        ordinal=0,
        text=text,
        title=doc_id,
        source_type=source_type,
        source_ref=doc_id,
        tlp=tlp,
    )


def _corpus() -> tuple[PlacedEmbedder, list[Document]]:
    embedder = PlacedEmbedder(
        {
            "process injection body": [1.0, 0.0, 0.0],
            "credential dumping body": [0.0, 1.0, 0.0],
        }
    )
    documents = [
        _document(
            "T1055", "T1055", "process injection body", "T1055 Process Injection (technique)"
        ),
        _document(
            "T1003", "T1003", "credential dumping body", "T1003 OS Credential Dumping (technique)"
        ),
    ]
    return embedder, documents


def test_a_public_chunk_is_matched_back_to_its_own_document() -> None:
    embedder, documents = _corpus()
    corpus, reference = build_reference(documents, embedder)
    target = _chunk("T1055#0", "T1055", "process injection body")

    report = reidentify(
        [target], {target.id: np.array([0.9, 0.1, 0.0], np.float32)}, corpus, reference
    )

    assert report.rows[0].correct is True
    assert report.top1_accuracy == 1.0


def test_a_wrong_nearest_neighbour_is_not_counted_as_recognised() -> None:
    embedder, documents = _corpus()
    corpus, reference = build_reference(documents, embedder)
    target = _chunk("T1055#0", "T1055", "process injection body")

    report = reidentify(
        [target], {target.id: np.array([0.0, 1.0, 0.0], np.float32)}, corpus, reference
    )

    assert report.rows[0].predicted_doc_id == "T1003"
    assert report.rows[0].correct is False
    assert report.top1_accuracy == 0.0


def test_a_confidential_note_is_scored_on_topic_disclosure_not_recognition() -> None:
    """Nothing public matches the note, but the neighbour still names its subject."""
    embedder, documents = _corpus()
    corpus, reference = build_reference(documents, embedder)
    note = _chunk(
        "INT-1#0",
        "INT-1",
        "Incident review: the intruder relied on process injection against the treasury host",
        source_type=SourceType.INTERNAL_NOTE,
        tlp=TLP.RED,
    )

    report = reidentify([note], {note.id: np.array([1.0, 0.0, 0.0], np.float32)}, corpus, reference)

    row = report.rows[0]
    assert row.in_reference is False
    assert row.correct is False
    assert row.topic_hit is True
    assert report.topic_disclosure_rate == 1.0


def test_topic_disclosure_does_not_require_the_note_to_quote_the_identifier() -> None:
    """Analysts write prose; scoring only literal ids reports zero disclosure."""
    embedder, documents = _corpus()
    corpus, reference = build_reference(documents, embedder)
    note = _chunk(
        "INT-1#0",
        "INT-1",
        "The actor performed process injection into a signed binary",
        source_type=SourceType.INTERNAL_NOTE,
    )

    report = reidentify([note], {note.id: np.array([1.0, 0.0, 0.0], np.float32)}, corpus, reference)

    assert report.rows[0].ref_hit is False
    assert report.rows[0].topic_hit is True


def test_a_confidential_note_whose_neighbour_misses_its_subject_is_not_a_hit() -> None:
    embedder, documents = _corpus()
    corpus, reference = build_reference(documents, embedder)
    note = _chunk(
        "INT-2#0",
        "INT-2",
        "Quarterly budget discussion with no technique named",
        source_type=SourceType.INTERNAL_NOTE,
        tlp=TLP.AMBER,
    )

    report = reidentify([note], {note.id: np.array([1.0, 0.0, 0.0], np.float32)}, corpus, reference)

    assert report.rows[0].topic_hit is False
    assert report.topic_disclosure_rate == 0.0


def test_the_two_populations_are_scored_separately() -> None:
    """Public chunks and confidential ones must never share a denominator."""
    embedder, documents = _corpus()
    corpus, reference = build_reference(documents, embedder)
    public = _chunk("T1055#0", "T1055", "process injection body")
    note = _chunk(
        "INT-1#0", "INT-1", "about process injection", source_type=SourceType.INTERNAL_NOTE
    )
    stolen = {
        public.id: np.array([1.0, 0.0, 0.0], np.float32),
        note.id: np.array([1.0, 0.0, 0.0], np.float32),
    }

    report = reidentify([public, note], stolen, corpus, reference)

    assert report.recognisable == 1
    assert report.unrecognisable == 1
    assert report.top1_accuracy == 1.0
    assert report.topic_disclosure_rate == 1.0


def test_topic_terms_drop_the_identifier_and_the_kind_marker() -> None:
    """Otherwise "technique" or "critical" in a note would read as disclosure."""
    assert topic_terms("T1003.001 LSASS Memory (technique)", "T1003.001") == ["lsass", "memory"]


def test_a_cve_title_yields_no_topic_terms() -> None:
    """An id and a severity name the subject only to someone who looks it up."""
    assert topic_terms("CVE-2026-26125 (CRITICAL)", "CVE-2026-26125") == []


def test_topic_terms_exclude_bare_numbers() -> None:
    """A note dated 2026-02-11 must not match every CVE published in 2026."""
    assert "2026" not in topic_terms("CVE-2026-26125 Apache Struts flaw", "CVE-2026-26125")


def test_the_two_disclosure_metrics_are_reported_separately() -> None:
    embedder, documents = _corpus()
    corpus, reference = build_reference(documents, embedder)
    quoting = _chunk(
        "INT-1#0", "INT-1", "the note names T1055 outright", source_type=SourceType.INTERNAL_NOTE
    )

    report = reidentify(
        [quoting], {quoting.id: np.array([1.0, 0.0, 0.0], np.float32)}, corpus, reference
    )

    assert report.ref_hits == 1
    assert report.rows[0].topic_hit is False


def test_the_reference_is_built_from_documents_not_chunks() -> None:
    """Matching stolen chunk vectors against the chunks they came from measures nothing."""
    embedder, documents = _corpus()

    corpus, reference = build_reference(documents, embedder)

    assert corpus.doc_ids == ["T1055", "T1003"]
    assert reference.shape == (2, 3)


def test_reference_embedding_is_batched() -> None:
    embedder = PlacedEmbedder({})
    documents = [_document(f"D{i}", f"D{i}", f"text {i}") for i in range(5)]

    build_reference(documents, embedder, batch_size=2)

    assert embedder.batches == [2, 2, 1]


def test_an_empty_reference_corpus_yields_an_empty_report() -> None:
    embedder = PlacedEmbedder({})
    corpus, reference = build_reference([], embedder)
    target = _chunk("T1055#0", "T1055", "body")

    report = reidentify([target], {target.id: np.ones(3, np.float32)}, corpus, reference)

    assert report.rows == []
    assert report.top1_accuracy == 0.0


def test_targets_with_no_stolen_vector_are_skipped() -> None:
    embedder, documents = _corpus()
    corpus, reference = build_reference(documents, embedder)
    target = _chunk("T1055#0", "T1055", "process injection body")

    report = reidentify([target], {}, corpus, reference)

    assert report.rows == []


def test_a_zero_vector_in_the_reference_does_not_win_every_match() -> None:
    """Normalising a zero row would produce nan and take the argmax."""
    embedder = PlacedEmbedder({"real body": [1.0, 0.0, 0.0]})
    documents = [_document("EMPTY", "EMPTY", "unknown"), _document("T1055", "T1055", "real body")]
    corpus, reference = build_reference(documents, embedder)
    target = _chunk("T1055#0", "T1055", "real body")

    report = reidentify(
        [target], {target.id: np.array([1.0, 0.0, 0.0], np.float32)}, corpus, reference
    )

    assert report.rows[0].predicted_doc_id == "T1055"
