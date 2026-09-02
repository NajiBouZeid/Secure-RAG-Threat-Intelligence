"""Sentence-transformers embedding backend.

Model loading is deferred until the first embed call so that constructing a
pipeline -- which the CLI and the tests both do eagerly -- does not pull ~100 MB
off HuggingFace as a side effect of parsing a config.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from threatrag.domain.types import Matrix, Vector

if TYPE_CHECKING:  # pragma: no cover - import cost only paid at runtime
    from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model behind the ``Embedder`` protocol."""

    def __init__(
        self,
        name: str,
        model_id: str,
        dim: int,
        *,
        normalize: bool = True,
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        self._name = name
        self._model_id = model_id
        self._dim = dim
        self._normalize = normalize
        self._device = device
        self._batch_size = batch_size
        self._model: SentenceTransformer | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_id(self) -> str:
        return self._model_id

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self._model_id, device=self._device)
            reported = model.get_sentence_embedding_dimension()
            if reported != self._dim:
                raise ValueError(
                    f"Config declares dim={self._dim} for {self._model_id!r} but the model "
                    f"reports {reported}. Fix configs/base.yaml -- a wrong dim silently "
                    f"corrupts the collection schema."
                )
            self._model = model
        return self._model

    def _encode(self, texts: Sequence[str], *, progress: bool = False) -> Matrix:
        vectors = self._load().encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=progress,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        return self._encode(texts, progress=len(texts) > 512)

    def embed_query(self, text: str) -> Vector:
        vector: Vector = self._encode([text])[0]
        return vector
