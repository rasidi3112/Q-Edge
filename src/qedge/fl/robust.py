"""Aggregation rules that limit the influence of malicious clients.

All rules implement `qedge.fl.strategies.Aggregator`. They need each
client's update in the clear, so they cannot run on top of secure
aggregation (`qedge.fl.secagg`), which reveals only the sum of the updates.

- ``median``: coordinate-wise median of the client weights.
- ``trimmed_mean``: coordinate-wise mean after dropping the ``beta``
  fraction of largest and smallest values. Both are from Yin et al.,
  "Byzantine-Robust Distributed Learning: Towards Optimal Statistical
  Rates", 2018, arXiv:1803.01498.
- ``krum`` and ``multi_krum``: keep the update(s) closest to their
  ``n - f - 2`` nearest neighbours (Blanchard et al., "Machine Learning with
  Adversaries: Byzantine Tolerant Gradient Descent", NeurIPS 2017).
- ``norm_clip``: clip every update ``w_i - w_global`` to the median update
  norm, then FedAvg (Sun et al., "Can You Really Backdoor Federated
  Learning?", 2019, arXiv:1911.07963).
- ``bounded``: FedAvg with each claimed sample count capped at ``cap``
  times the median claim, which limits inflated ``n_samples``.

Median, trimmed mean and Krum give every client one vote and ignore sample
counts.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from qedge.fl.strategies import Weights, fedavg

DEFENSES = ("median", "trimmed_mean", "krum", "multi_krum", "norm_clip", "bounded")


def _stack(updates: Sequence[tuple[Weights, int]]) -> tuple[list[str], NDArray[np.float64]]:
    if not updates:
        raise ValueError("Need at least one update")
    names = list(updates[0][0])
    if any(list(w) != names for w, _ in updates):
        raise ValueError("All updates must have the same tensor names")
    flat = np.stack(
        [np.concatenate([np.ravel(w[n]) for n in names]).astype(np.float64) for w, _ in updates]
    )
    return names, flat


def _unstack(
    names: list[str], vector: NDArray[np.float64], like: Weights
) -> dict[str, NDArray[Any]]:
    result, offset = {}, 0
    for name in names:
        ref = np.asarray(like[name])
        result[name] = vector[offset : offset + ref.size].reshape(ref.shape).astype(ref.dtype)
        offset += ref.size
    return result


def coordinate_median(updates: Sequence[tuple[Weights, int]]) -> dict[str, NDArray[Any]]:
    """Coordinate-wise median of the client weights."""
    names, flat = _stack(updates)
    return _unstack(names, np.median(flat, axis=0), updates[0][0])


def trimmed_mean(
    updates: Sequence[tuple[Weights, int]], beta: float = 0.2
) -> dict[str, NDArray[Any]]:
    """Coordinate-wise mean without the ``floor(beta * n)`` largest and smallest values.

    Raises:
        ValueError: If ``beta`` is not in [0, 0.5) or nothing would remain.
    """
    if not 0 <= beta < 0.5:
        raise ValueError("beta must be in [0, 0.5)")
    names, flat = _stack(updates)
    k = int(np.floor(beta * len(flat)))
    if len(flat) - 2 * k < 1:
        raise ValueError("trimming would remove every update")
    ordered = np.sort(flat, axis=0)
    return _unstack(names, ordered[k : len(flat) - k].mean(axis=0), updates[0][0])


def krum_scores(flat: NDArray[np.float64], f: int) -> NDArray[np.float64]:
    """Sum of squared distances from each update to its ``n - f - 2`` nearest others."""
    n = len(flat)
    distances = np.sum((flat[:, None, :] - flat[None, :, :]) ** 2, axis=-1)
    neighbours = n - f - 2
    return np.array([np.sort(np.delete(distances[i], i))[:neighbours].sum() for i in range(n)])


def multi_krum(
    updates: Sequence[tuple[Weights, int]], f: int, m: int | None = None
) -> dict[str, NDArray[Any]]:
    """Average of the ``m`` updates with the lowest Krum score; ``m = 1`` is Krum.

    Args:
        updates: ``(weights, n_samples)`` pairs; sample counts are ignored.
        f: Number of Byzantine clients the rule tolerates; needs ``n > 2f + 2``.
        m: Updates to average; defaults to ``n - f``.

    Raises:
        ValueError: If ``n <= 2f + 2`` or ``m`` is out of range.
    """
    names, flat = _stack(updates)
    n = len(flat)
    if n <= 2 * f + 2:
        raise ValueError(f"Krum needs n > 2f + 2; got n={n}, f={f}")
    m = n - f if m is None else m
    if not 1 <= m <= n:
        raise ValueError("m must be between 1 and n")
    chosen = np.argsort(krum_scores(flat, f), kind="stable")[:m]
    return _unstack(names, flat[chosen].mean(axis=0), updates[0][0])


def norm_clipped_fedavg(
    global_weights: Weights, updates: Sequence[tuple[Weights, int]], clip: float | None = None
) -> dict[str, NDArray[Any]]:
    """FedAvg of updates whose deltas are clipped to ``clip`` (default: median delta norm)."""
    names, flat = _stack(updates)
    origin = np.concatenate([np.ravel(global_weights[n]) for n in names]).astype(np.float64)
    deltas = flat - origin
    norms = np.linalg.norm(deltas, axis=1)
    bound = float(np.median(norms)) if clip is None else clip
    scale = np.minimum(1.0, bound / np.maximum(norms, 1e-12))
    clipped = [
        (_unstack(names, origin + delta * s, updates[0][0]), n)
        for delta, s, (_, n) in zip(deltas, scale, updates, strict=True)
    ]
    return fedavg(clipped)


def bounded_fedavg(
    updates: Sequence[tuple[Weights, int]], cap: float = 3.0
) -> dict[str, NDArray[Any]]:
    """FedAvg with sample counts capped at ``cap`` times the median claimed count.

    Raises:
        ValueError: If ``cap`` is below 1.
    """
    if cap < 1:
        raise ValueError("cap must be at least 1")
    counts = np.array([n for _, n in updates], dtype=np.float64)
    bound = cap * float(np.median(counts))
    return fedavg([(w, int(min(n, bound))) for w, n in updates])


class RobustAggregator:
    """Stateless `Aggregator` for one of `DEFENSES`.

    Args:
        name: One of `DEFENSES`.
        **options: ``beta`` (trimmed_mean), ``f`` and ``m`` (krum,
            multi_krum), ``clip`` (norm_clip), ``cap`` (bounded).

    Raises:
        ValueError: If ``name`` is unknown or an option does not apply.
    """

    _OPTIONS: ClassVar[dict[str, set[str]]] = {
        "median": set(),
        "trimmed_mean": {"beta"},
        "krum": {"f"},
        "multi_krum": {"f", "m"},
        "norm_clip": {"clip"},
        "bounded": {"cap"},
    }

    def __init__(self, name: str, **options: Any) -> None:
        if name not in DEFENSES:
            raise ValueError(f"Unknown defense {name!r}; expected {DEFENSES}")
        extra = set(options) - self._OPTIONS[name]
        if extra:
            raise ValueError(f"Options {sorted(extra)} do not apply to {name}")
        self.name = name
        self.options = options

    def step(
        self, global_weights: Weights, updates: Sequence[tuple[Weights, int]]
    ) -> dict[str, NDArray[Any]]:
        """Return the next global weights."""
        n = len(updates)
        if self.name == "median":
            return coordinate_median(updates)
        if self.name == "trimmed_mean":
            return trimmed_mean(updates, **self.options)
        if self.name in ("krum", "multi_krum"):
            f = int(self.options.get("f", max((n - 3) // 2, 0)))
            m = 1 if self.name == "krum" else self.options.get("m")
            return multi_krum(updates, f, m)
        if self.name == "norm_clip":
            return norm_clipped_fedavg(global_weights, updates, **self.options)
        return bounded_fedavg(updates, **self.options)


SERVER_RULES = ("fedavg", "median", "trimmed_mean", "krum", "multi_krum", "bounded")


def server_rule(name: str) -> Callable[[Sequence[tuple[Weights, int]]], dict[str, NDArray[Any]]]:
    """Aggregation for a server round, which has no previous global model.

    ``norm_clip`` is not offered because it clips against the previous
    global weights.

    Raises:
        ValueError: If ``name`` is not one of `SERVER_RULES`.
    """
    if name not in SERVER_RULES:
        raise ValueError(f"Unknown server aggregation {name!r}; expected {SERVER_RULES}")
    if name == "fedavg":
        return fedavg
    aggregator = RobustAggregator(name)
    return lambda updates: aggregator.step(updates[0][0], updates)
