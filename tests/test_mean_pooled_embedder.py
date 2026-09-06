"""The embedding convention the public GTR corrector was trained against.

Loading the real encoder is a 220 MB download, so what is pinned here is the
part that was actually wrong: the pooling arithmetic, and the fact that the
config resolves gtr-base to this backend rather than to sentence-transformers.
The equivalence to vec2text's own code was checked against the live model
during M5 -- max absolute difference 0.0 -- and the cosine between a
sentence-transformers vector and this one was 0.018, which is why the
distinction is not cosmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from threatrag.config import load_config
from threatrag.index.embedders.mean_pooled import MeanPooledEncoderEmbedder, _mean_pool
from threatrag.index.embedders.sentence_transformer import SentenceTransformerEmbedder

torch = pytest.importorskip("torch")


def test_mean_pool_averages_only_unmasked_positions() -> None:
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])
    mask = torch.tensor([[1, 1, 0]])

    pooled = _mean_pool(hidden, mask)

    assert pooled.tolist() == [[2.0, 2.0]]


def test_mean_pool_does_not_normalise() -> None:
    """sentence-transformers L2-normalises here; the corrector was not trained on that."""
    hidden = torch.tensor([[[3.0, 4.0]]])
    mask = torch.tensor([[1]])

    pooled = _mean_pool(hidden, mask)

    assert float(torch.linalg.norm(pooled)) == pytest.approx(5.0)


def test_mean_pool_handles_a_single_live_token() -> None:
    hidden = torch.tensor([[[2.0, 6.0], [0.0, 0.0]]])
    mask = torch.tensor([[1, 0]])

    assert _mean_pool(hidden, mask).tolist() == [[2.0, 6.0]]


def test_an_empty_batch_returns_the_right_shape_without_loading_a_model() -> None:
    embedder = MeanPooledEncoderEmbedder("gtr-base", "does-not-exist", 768)

    result = embedder.embed_documents([])

    assert result.shape == (0, 768)
    assert result.dtype == np.float32


def test_gtr_resolves_to_the_corrector_convention_not_sentence_transformers() -> None:
    """The whole GTR inversion result depends on this one config line."""
    from threatrag import factory

    embedder = factory.build_embedder(load_config(None, None), "gtr-base")

    assert isinstance(embedder, MeanPooledEncoderEmbedder)
    assert embedder.max_tokens is None


def test_the_retrieval_encoder_is_untouched_by_that_change() -> None:
    from threatrag import factory

    embedder = factory.build_embedder(load_config(None, None), None)

    assert isinstance(embedder, SentenceTransformerEmbedder)
    assert embedder.name == "all-minilm"


def test_max_tokens_can_be_overridden_for_one_call() -> None:
    """The 32-token control bundle cannot declare a second named vector."""
    from threatrag import factory

    control = factory.build_embedder(load_config(None, None), "gtr-base", max_tokens=32)

    assert control.max_tokens == 32
    assert control.name == "gtr-base"


def test_an_unknown_backend_is_rejected() -> None:
    from threatrag import factory

    config = load_config(None, None)
    config.embedding.models["gtr-base"].backend = "nonsense"

    with pytest.raises(ValueError, match="Unsupported embedding backend"):
        factory.build_embedder(config, "gtr-base")
