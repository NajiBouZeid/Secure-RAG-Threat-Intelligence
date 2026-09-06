"""Export of the attack bundle and the answer key.

The split is the thing under test: the files that leave the machine must carry
vectors and ids only. Handing the reconstruction step the source text would
make any reconstruction rate it produced meaningless.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from threatrag.domain.models import TLP, Chunk, SourceType
from threatrag.domain.types import Matrix, Vector
from threatrag.security.inversion.export import (
    IDS_FILE,
    TRUTH_FILE,
    VECTORS_FILE,
    export_control,
    export_targets,
    load_truth,
)
from threatrag.security.inversion.sample import SampleReport


def _chunk(index: int, text: str = "secret body", tlp: TLP = TLP.CLEAR) -> Chunk:
    return Chunk(
        id=f"c-{index}#0",
        doc_id=f"c-{index}",
        ordinal=0,
        text=text,
        title=f"title {index}",
        source_type=SourceType.INTERNAL_NOTE,
        source_ref=f"INT-{index}",
        tlp=tlp,
        metadata={"secret_terms": ["41 repositories"]},
    )


class VectorStub:
    def __init__(self, vectors: dict[str, Vector]) -> None:
        self.vectors = vectors

    def get_vectors(self, vector_name: str, chunk_ids: Sequence[str]) -> dict[str, Vector]:
        return {cid: self.vectors[cid] for cid in chunk_ids if cid in self.vectors}


def _sample(chunks: list[Chunk]) -> SampleReport:
    return SampleReport(seed=7, requested=len(chunks), chunks=chunks)


def test_the_uploaded_bundle_carries_no_source_text(tmp_path: Path) -> None:
    chunks = [_chunk(0, text="TLP:RED payment approval matrix")]
    store = VectorStub({chunks[0].id: np.ones(4, dtype=np.float32)})

    export_targets(store, _sample(chunks), vector_name="gtr-base", out_dir=tmp_path)

    uploaded = (tmp_path / IDS_FILE).read_text(encoding="utf-8")
    assert "payment approval matrix" not in uploaded
    assert json.loads(uploaded) == [chunks[0].id]


def test_vectors_are_written_in_the_order_of_the_ids(tmp_path: Path) -> None:
    chunks = [_chunk(0), _chunk(1)]
    store = VectorStub(
        {
            chunks[0].id: np.array([1, 0, 0, 0], dtype=np.float32),
            chunks[1].id: np.array([0, 1, 0, 0], dtype=np.float32),
        }
    )

    export_targets(store, _sample(chunks), vector_name="gtr-base", out_dir=tmp_path)

    ids = json.loads((tmp_path / IDS_FILE).read_text(encoding="utf-8"))
    matrix = np.load(tmp_path / VECTORS_FILE)
    assert ids == [chunks[0].id, chunks[1].id]
    assert matrix[ids.index(chunks[1].id)].tolist() == [0, 1, 0, 0]


def test_chunks_without_that_named_vector_are_reported_not_dropped_silently(
    tmp_path: Path,
) -> None:
    """Most of the index carries MiniLM only; a partial backfill must be visible."""
    chunks = [_chunk(0), _chunk(1)]
    store = VectorStub({chunks[0].id: np.ones(4, dtype=np.float32)})

    manifest = export_targets(store, _sample(chunks), vector_name="gtr-base", out_dir=tmp_path)

    assert manifest.exported == 1
    assert manifest.missing == [chunks[1].id]


def test_the_answer_key_keeps_the_text_and_the_secret_terms(tmp_path: Path) -> None:
    chunks = [_chunk(0, text="INT-0 TLP:RED 41 repositories payment approval matrix", tlp=TLP.RED)]
    store = VectorStub({chunks[0].id: np.ones(4, dtype=np.float32)})

    export_targets(store, _sample(chunks), vector_name="gtr-base", out_dir=tmp_path)

    truth = load_truth(tmp_path / TRUTH_FILE)
    record = truth[chunks[0].id]
    assert record["text"] == "INT-0 TLP:RED 41 repositories payment approval matrix"
    assert record["tlp"] == "red"
    assert record["secret_terms"] == ["INT-0", "41 repositories"]


def test_the_manifest_records_the_sampling_method(tmp_path: Path) -> None:
    chunks = [_chunk(0)]
    store = VectorStub({chunks[0].id: np.ones(4, dtype=np.float32)})
    sample = SampleReport(
        seed=99,
        requested=1,
        population={"internal_note": 34},
        selected={"internal_note": 1},
        chunks=chunks,
    )

    manifest = export_targets(store, sample, vector_name="gtr-base", out_dir=tmp_path)

    assert manifest.seed == 99
    assert manifest.population == {"internal_note": 34}
    assert manifest.dim == 4
    assert manifest.upload_files == (VECTORS_FILE, IDS_FILE)


def test_an_empty_sample_still_writes_a_readable_bundle(tmp_path: Path) -> None:
    manifest = export_targets(VectorStub({}), _sample([]), vector_name="gtr-base", out_dir=tmp_path)

    assert manifest.exported == 0
    assert np.load(tmp_path / VECTORS_FILE).shape[0] == 0
    assert load_truth(tmp_path / TRUTH_FILE) == {}


class TruncatingStub:
    """Stands in for the mean-pooled encoder: records what it was asked to embed."""

    name = "gtr-base"
    max_tokens = 4

    def __init__(self) -> None:
        self.seen: list[str] = []

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        self.seen = list(texts)
        return np.ones((len(texts), 3), dtype=np.float32)

    def tokenized_prefix(self, text: str) -> str:
        return " ".join(text.split()[: self.max_tokens])


def test_the_control_answer_key_holds_the_truncated_prefix(tmp_path: Path) -> None:
    """Scoring a 32-token vector against 512 characters marks it wrong for
    omitting words its vector never carried."""
    chunk = _chunk(0, text="alpha beta gamma delta epsilon zeta")

    export_control(
        _sample([chunk]), TruncatingStub(), out_dir=tmp_path, variant="len32_unnormalized"
    )

    record = load_truth(tmp_path / TRUTH_FILE)[chunk.id]
    assert record["text"] == "alpha beta gamma delta"


def test_the_control_embeds_the_original_text_not_the_decoded_prefix(tmp_path: Path) -> None:
    """Embedding the decoded prefix pushes the text through a lossy
    decode-and-re-encode round trip, so the bundle would encode subtly
    different text from the one the index holds."""
    chunk = _chunk(0, text="alpha beta gamma delta epsilon zeta")
    embedder = TruncatingStub()

    export_control(_sample([chunk]), embedder, out_dir=tmp_path, variant="len32_unnormalized")

    assert embedder.seen == ["alpha beta gamma delta epsilon zeta"]


def test_control_secret_terms_drop_out_when_truncated_away(tmp_path: Path) -> None:
    """A secret past the token budget is not in the vector, so recovering it
    would be impossible and counting it would understate the attack."""
    chunk = _chunk(0, text="alpha beta gamma delta 41 repositories").model_copy(
        update={"metadata": {"secret_terms": ["41 repositories"]}}
    )

    export_control(
        _sample([chunk]), TruncatingStub(), out_dir=tmp_path, variant="len32_unnormalized"
    )

    assert load_truth(tmp_path / TRUTH_FILE)[chunk.id]["secret_terms"] == []


def test_the_control_manifest_records_the_token_budget(tmp_path: Path) -> None:
    manifest = export_control(
        _sample([_chunk(0)]), TruncatingStub(), out_dir=tmp_path, variant="len32_unnormalized"
    )

    assert manifest.max_tokens == 4
    assert manifest.exported == 1


def test_each_bundle_records_which_variant_it_is(tmp_path: Path) -> None:
    """Three bundles differ in ways that mean different things; a score is
    uninterpretable without knowing which one produced it."""
    manifest = export_control(
        _sample([_chunk(0)]), TruncatingStub(), out_dir=tmp_path, variant="full_unnormalized"
    )

    assert manifest.variant == "full_unnormalized"


def test_the_stored_bundle_is_labelled_as_stored(tmp_path: Path) -> None:
    chunk = _chunk(0)
    store = VectorStub({chunk.id: np.ones(4, dtype=np.float32)})

    manifest = export_targets(store, _sample([chunk]), vector_name="gtr-base", out_dir=tmp_path)

    assert manifest.variant == "stored"
