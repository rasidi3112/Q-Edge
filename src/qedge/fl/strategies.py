"""Aggregation rules as pure functions over named weight arrays."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

Weights = Mapping[str, NDArray[Any]]


class Aggregator(Protocol):
    """Anything that turns the global weights and client updates into new weights."""

    def step(
        self, global_weights: Weights, updates: Sequence[tuple[Weights, int]]
    ) -> dict[str, NDArray[Any]]:
        """Return the next global weights."""
        ...


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


@dataclass(frozen=True)
class AdaptiveState:
    """First and second moment estimates kept by FedAdam and FedYogi."""

    m: dict[str, NDArray[np.float64]]
    v: dict[str, NDArray[np.float64]]


def _adaptive_step(
    global_weights: Weights,
    updates: Sequence[tuple[Weights, int]],
    state: AdaptiveState | None,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    tau: float,
    yogi: bool,
) -> tuple[dict[str, NDArray[Any]], AdaptiveState]:
    average = fedavg(updates)
    if state is None:
        state = AdaptiveState(
            m={n: np.zeros(np.shape(w), np.float64) for n, w in global_weights.items()},
            v={n: np.full(np.shape(w), tau**2, np.float64) for n, w in global_weights.items()},
        )
    new_weights: dict[str, NDArray[Any]] = {}
    m_next: dict[str, NDArray[np.float64]] = {}
    v_next: dict[str, NDArray[np.float64]] = {}
    for name, current in global_weights.items():
        delta = average[name].astype(np.float64) - np.asarray(current, np.float64)
        m = beta1 * state.m[name] + (1 - beta1) * delta
        squared = delta**2
        if yogi:
            v = state.v[name] - (1 - beta2) * squared * np.sign(state.v[name] - squared)
        else:
            v = beta2 * state.v[name] + (1 - beta2) * squared
        step = np.asarray(current, np.float64) + lr * m / (np.sqrt(v) + tau)
        new_weights[name] = step.astype(np.asarray(current).dtype)
        m_next[name], v_next[name] = m, v
    return new_weights, AdaptiveState(m_next, v_next)


def fedadam(
    global_weights: Weights,
    updates: Sequence[tuple[Weights, int]],
    state: AdaptiveState | None,
    *,
    lr: float = 0.05,
    beta1: float = 0.9,
    beta2: float = 0.99,
    tau: float = 1e-3,
) -> tuple[dict[str, NDArray[Any]], AdaptiveState]:
    """One FedAdam server step.

    Reddi et al., "Adaptive Federated Optimization", 2020, arXiv:2003.00295
    (Algorithm 2). The pseudo-gradient is the FedAvg aggregate minus the
    current global weights; ``m`` and ``v`` persist across rounds, there is no
    bias correction, and ``v`` starts at ``tau**2``.

    Returns:
        The new global weights and the state to pass to the next round.
    """
    return _adaptive_step(
        global_weights, updates, state, lr=lr, beta1=beta1, beta2=beta2, tau=tau, yogi=False
    )


def fedyogi(
    global_weights: Weights,
    updates: Sequence[tuple[Weights, int]],
    state: AdaptiveState | None,
    *,
    lr: float = 0.05,
    beta1: float = 0.9,
    beta2: float = 0.99,
    tau: float = 1e-3,
) -> tuple[dict[str, NDArray[Any]], AdaptiveState]:
    """One FedYogi server step: FedAdam with Yogi's additive ``v`` update.

    ``v <- v - (1 - beta2) * delta**2 * sign(v - delta**2)`` (Reddi et al. 2020).
    """
    return _adaptive_step(
        global_weights, updates, state, lr=lr, beta1=beta1, beta2=beta2, tau=tau, yogi=True
    )


class ServerStrategy:
    """Stateful wrapper so callers can treat every strategy the same way.

    Args:
        name: One of `SERVER_STRATEGIES`. FedProx is not here: it changes the
            client loss (`qedge.fl.training.TrainConfig.mu`) and aggregates
            with ``fedavg``.
        **options: Keyword options for `fedadam` or `fedyogi`.

    Raises:
        ValueError: If ``name`` is unknown or options are given for FedAvg.
    """

    def __init__(self, name: str, **options: float) -> None:
        if name not in SERVER_STRATEGIES:
            raise ValueError(f"Unknown server strategy {name!r}; expected {SERVER_STRATEGIES}")
        if name == "fedavg" and options:
            raise ValueError("FedAvg takes no options")
        self.name = name
        self._options = options
        self._state: AdaptiveState | None = None

    def step(
        self, global_weights: Weights, updates: Sequence[tuple[Weights, int]]
    ) -> dict[str, NDArray[Any]]:
        """Return the next global weights from the current ones and the updates."""
        if self.name == "fedavg":
            return fedavg(updates)
        rule = fedadam if self.name == "fedadam" else fedyogi
        weights, self._state = rule(global_weights, updates, self._state, **self._options)
        return weights


SERVER_STRATEGIES = ("fedavg", "fedadam", "fedyogi")
