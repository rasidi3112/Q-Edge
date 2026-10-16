"""Aggregation rules as pure functions over named weight arrays."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

Weights = Mapping[str, NDArray[Any]]


def fedavg(updates: Sequence[tuple[Weights, int]]) -> dict[str, NDArray[Any]]:
    """Sample-weighted average of client weights (FedAvg).

    McMahan et al., "Communication-Efficient Learning of Deep Networks from
    Decentralized Data", 2017, arXiv:1602.05629.

    Args:
        updates: ``(weights, n_samples)`` pairs; all weights share names,
            shapes and dtypes.

    Returns:
        The weighted mean per tensor, in the dtype of the inputs.

    Raises:
        ValueError: If ``updates`` is empty, the total sample count is zero,
            or tensor names differ.
    """
    if not updates:
        raise ValueError("FedAvg needs at least one update")
    total = sum(n for _, n in updates)
    if total <= 0:
        raise ValueError("FedAvg needs a positive total sample count")
    names = list(updates[0][0])
    if any(list(weights) != names for weights, _ in updates):
        raise ValueError("All updates must have the same tensor names")
    result: dict[str, NDArray[Any]] = {}
    for name in names:
        dtype = updates[0][0][name].dtype
        acc = np.zeros(updates[0][0][name].shape, dtype=np.float64)
        for weights, n in updates:
            acc += (n / total) * weights[name].astype(np.float64)
        result[name] = acc.astype(dtype)
    return result
