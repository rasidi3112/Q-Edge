"""Simulated malicious clients.

- ``label_flip``: train on ``n_classes - 1 - y`` instead of ``y``.
- ``scaling``: train on flipped labels, then send ``scale * delta`` so the
  poisoned model survives averaging (model replacement; Bagdasaryan et al.,
  "How To Backdoor Federated Learning", 2018, arXiv:1807.00459). Default
  ``scale`` is the number of clients.
- ``random_noise``: send ``delta ~ N(0, scale^2)`` per coordinate, without
  training. Default ``scale`` 1.
- ``fake_samples``: train on flipped labels like ``label_flip`` and claim
  ``scale`` times the real sample count, so FedAvg gives the poisoned model
  most of the weight. Default ``scale`` 100.

Attackers are a fixed set of clients drawn once per run from the seed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

ATTACKS = ("label_flip", "scaling", "random_noise", "fake_samples")
_DEFAULT_SCALE = {"label_flip": None, "random_noise": 1.0, "fake_samples": 100.0}


@dataclass(frozen=True)
class Attack:
    """One attack applied by a fixed fraction of clients.

    Raises:
        ValueError: If ``name`` is unknown, ``fraction`` is outside [0, 1) or
            ``scale`` is given for ``label_flip``.
    """

    name: str
    fraction: float
    scale: float | None = None

    def __post_init__(self) -> None:
        if self.name not in ATTACKS:
            raise ValueError(f"Unknown attack {self.name!r}; expected {ATTACKS}")
        if not 0 <= self.fraction < 1:
            raise ValueError("attack fraction must be in [0, 1)")
        if self.name == "label_flip" and self.scale is not None:
            raise ValueError("label_flip takes no scale")

    def attackers(self, n_clients: int, seed: int) -> frozenset[int]:
        """``round(fraction * n_clients)`` client indices drawn with ``seed``."""
        k = round(self.fraction * n_clients)
        rng = np.random.default_rng([seed, 7])
        return frozenset(int(i) for i in rng.choice(n_clients, size=k, replace=False))

    def resolved_scale(self, n_clients: int) -> float | None:
        """``scale`` or its default for this attack."""
        if self.scale is not None:
            return self.scale
        return float(n_clients) if self.name == "scaling" else _DEFAULT_SCALE[self.name]

    def _factor(self, n_clients: int) -> float:
        scale = self.resolved_scale(n_clients)
        return 1.0 if scale is None else scale

    def labels(self, y: NDArray[np.int64], n_classes: int) -> NDArray[np.int64]:
        """Training labels the attacker uses."""
        return y if self.name == "random_noise" else n_classes - 1 - y

    def delta(
        self,
        delta: Mapping[str, NDArray[Any]],
        n_clients: int,
        rng: np.random.Generator,
    ) -> dict[str, NDArray[Any]]:
        """The update the attacker sends instead of its honest ``delta``."""
        scale = self._factor(n_clients)
        if self.name == "scaling":
            return {
                k: (np.asarray(v) * scale).astype(np.asarray(v).dtype) for k, v in delta.items()
            }
        if self.name == "random_noise":
            return {
                k: rng.normal(0.0, scale, np.shape(v)).astype(np.asarray(v).dtype)
                for k, v in delta.items()
            }
        return dict(delta)

    def claimed_samples(self, n_samples: int, n_clients: int) -> int:
        """The sample count the attacker reports."""
        if self.name == "fake_samples":
            return int(n_samples * self._factor(n_clients))
        return n_samples
