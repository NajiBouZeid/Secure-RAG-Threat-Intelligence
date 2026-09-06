"""Raw encoder hidden states, masked-mean pooled, unnormalised.

This exists to match one specific convention rather than to be a good encoder.
The public GTR corrector (`jxm/gtr__nq__32__correct`) was trained against
vec2text's own `gtr_base` embedder, which is:

    AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder
    -> last_hidden_state
    -> masked mean over tokens
    -> returned as-is

Note what is *not* there. sentence-transformers runs the same backbone and then
applies a 768->768 Dense projection and L2 normalisation, so its output and the
above are different vectors of the same width, living in different spaces. An
inversion attack fed the wrong one returns fluent nonsense and looks exactly
like an encoder that resists inversion -- which is the false negative this
module exists to prevent. The M5 bundle was originally exported the
sentence-transformers way and would have produced that result.

`max_tokens` matters for the same reason: the corrector was trained on
32-token sequences, so a bundle meant to measure the attack on its home turf
has to be tokenised the same way.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from threatrag.domain.types import Matrix, Vector

if TYPE_CHECKING:  # pragma: no cover - import cost only paid at runtime
    import torch


class MeanPooledEncoderEmbedder:
    """Wraps a transformers encoder behind the ``Embedder`` protocol."""

    def __init__(
        self,
        name: str,
        model_id: str,
        dim: int,
        *,
        max_tokens: int | None = None,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self._name = name
        self._model_id = model_id
        self._dim = dim
        self._max_tokens = max_tokens
        self._device = device
        self._batch_size = batch_size
        self._model: Any = None
        self._tokenizer: Any = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def max_tokens(self) -> int | None:
        return self._max_tokens

    def _load(self) -> tuple[Any, Any]:
        if self._model is None:
            import torch
            import transformers

            model = transformers.AutoModel.from_pretrained(self._model_id)
            # .encoder, matching vec2text: for a T5 checkpoint AutoModel returns
            # the full encoder-decoder and only the encoder produces the
            # embedding.
            encoder = getattr(model, "encoder", model)
            hidden = int(encoder.config.d_model if hasattr(encoder.config, "d_model") else 0)
            if hidden and hidden != self._dim:
                raise ValueError(
                    f"Config declares dim={self._dim} for {self._model_id!r} but the encoder "
                    f"reports {hidden}. A wrong dim silently corrupts the collection schema."
                )

            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            encoder = encoder.to(device).eval()
            self._model = encoder
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(self._model_id)
        return self._model, self._tokenizer

    def _encode(self, texts: Sequence[str]) -> Matrix:
        import torch

        encoder, tokenizer = self._load()
        device = next(encoder.parameters()).device
        blocks: list[Matrix] = []

        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self._max_tokens or tokenizer.model_max_length,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                hidden_state = encoder(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                ).last_hidden_state

            blocks.append(_mean_pool(hidden_state, encoded["attention_mask"]).cpu().numpy())

        if not blocks:
            return np.empty((0, self._dim), dtype=np.float32)
        return np.vstack(blocks).astype(np.float32)

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        return self._encode(texts)

    def embed_query(self, text: str) -> Vector:
        vector: Vector = self._encode([text])[0]
        return vector

    def tokenized_prefix(self, text: str) -> str:
        """The text the encoder actually saw, decoded back.

        A truncated bundle has to be scored against what was embedded, not
        against the full chunk, or the reconstruction is marked wrong for
        omitting words its vector never carried.
        """
        _, tokenizer = self._load()
        ids = tokenizer(
            text,
            truncation=True,
            max_length=self._max_tokens or tokenizer.model_max_length,
        )["input_ids"]
        return str(tokenizer.decode(ids, skip_special_tokens=True))


def _mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Masked mean over the sequence, exactly as vec2text computes it."""
    masked = hidden_states * attention_mask[..., None]
    pooled: torch.Tensor = masked.sum(dim=1) / attention_mask.sum(dim=1)[:, None]
    return pooled
