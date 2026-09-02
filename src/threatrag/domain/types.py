"""Shared numeric aliases.

``np.ndarray`` unparameterised defeats strict type checking, and spelling the
full generic at every call site is noise.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# A single embedding.
Vector = npt.NDArray[np.float32]

# A batch of embeddings, shape (n, dim).
Matrix = npt.NDArray[np.float32]
