"""Clipping and Gaussian noise for client updates.

This is the mechanism of DP-FedAvg-style training (clip each client delta to
L2 norm ``clip``, add N(0, (noise_multiplier * clip)^2) per coordinate). No
privacy accountant is implemented, so no epsilon is computed or reported.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray


def clip_and_noise(
    delta: Mapping[str, NDArray[Any]],
    *,
    clip: float,
    noise_multiplier: float,
    rng: np.random.Generator,
) -> tuple[dict[str, NDArray[Any]], float]:
    """Clip ``delta`` to global L2 norm ``clip``, then add Gaussian noise.

    Returns:
        The privatized delta and the L2 norm before clipping.

    Raises:
        ValueError: If ``clip`` is not positive or ``noise_multiplier`` negative.
    """
    if clip <= 0 or noise_multiplier < 0:
        raise ValueError("clip must be > 0 and noise_multiplier >= 0")
    norm = float(np.sqrt(sum(float(np.sum(np.square(v))) for v in delta.values())))
    scale = min(1.0, clip / norm) if norm > 0 else 1.0
    sigma = noise_multiplier * clip
    noisy = {
        name: (np.asarray(v, np.float64) * scale + rng.normal(0.0, sigma, np.shape(v))).astype(
            np.asarray(v).dtype
        )
        for name, v in delta.items()
    }
    return noisy, norm
