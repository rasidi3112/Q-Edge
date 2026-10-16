"""Top-k sparsification of client updates with error feedback.

Each client sends only the ``k`` largest-magnitude entries of
``delta + residual`` and keeps the rest as its new residual, so dropped
mass is sent in later rounds instead of lost. Karimireddy et al., "Error
Feedback Fixes SignSGD and other Gradient Compression Schemes", 2019,
arXiv:1901.09847. Compression applies to the update ``w_local - w_global``,
never to the weights themselves.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

# A kept entry costs a 4-byte index and a 4-byte float32 value.
BYTES_PER_KEPT_ENTRY = 8
BYTES_PER_DENSE_ENTRY = 4


def dense_bytes(weights: Mapping[str, NDArray[Any]]) -> int:
    """Bytes to send ``weights`` uncompressed as float32."""
    return BYTES_PER_DENSE_ENTRY * sum(int(np.size(w)) for w in weights.values())


class TopKCompressor:
    """Per-client top-k compressor with an error-feedback residual.

    Args:
        ratio: Fraction of entries kept per tensor, in (0, 1].

    Raises:
        ValueError: If ``ratio`` is outside (0, 1].
    """

    def __init__(self, ratio: float) -> None:
        if not 0 < ratio <= 1:
            raise ValueError("ratio must be in (0, 1]")
        self.ratio = ratio
        self._residual: dict[str, NDArray[np.float64]] = {}

    def compress(self, delta: Mapping[str, NDArray[Any]]) -> tuple[dict[str, NDArray[Any]], int]:
        """Sparsify ``delta`` plus the stored residual.

        Returns:
            The dense reconstruction the server will apply (zeros outside the
            kept entries) and the number of bytes sent.
        """
        sent: dict[str, NDArray[Any]] = {}
        nbytes = 0
        for name, value in delta.items():
            flat = np.asarray(value, np.float64).ravel() + self._residual.get(
                name, np.zeros(np.size(value))
            )
            k = max(1, round(self.ratio * flat.size))
            keep = np.argpartition(np.abs(flat), -k)[-k:]
            sparse = np.zeros_like(flat)
            sparse[keep] = flat[keep]
            self._residual[name] = flat - sparse
            sent[name] = sparse.reshape(np.shape(value)).astype(np.asarray(value).dtype)
            nbytes += BYTES_PER_KEPT_ENTRY * k
        return sent, nbytes
