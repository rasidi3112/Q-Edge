"""Trainability and expressivity diagnostics for the circuits in this package.

- `gradient_variance`: variance of one partial derivative over random
  weights. It shrinks exponentially with the qubit count for global cost
  functions (McClean et al., "Barren plateaus in quantum neural network
  training landscapes", 2018, arXiv:1803.11173).
- `kernel_concentration`: spread of off-diagonal fidelity-kernel values
  (Thanasilp et al., 2022, arXiv:2208.11060).
- `expressibility`: KL divergence between the distribution of state
  fidelities and the Haar distribution (Sim et al., "Expressibility and
  entangling capability of parameterized quantum circuits", 2019,
  arXiv:1905.10876). Sampling weights measures the ansatz; sampling inputs
  with fixed weights measures how far the embedding spreads the data, which
  is a different quantity.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pennylane as qml
from numpy.typing import NDArray

from qedge.quantum.circuits import variational_block, weight_shape
from qedge.quantum.kernels import QuantumKernel


def gradient_variance(
    n_qubits: int,
    n_layers: int,
    *,
    ansatz: str = "strongly_entangling",
    observable: str = "global",
    samples: int = 200,
    seed: int = 0,
) -> float:
    """Variance of ``d<O>/dw`` for the first RY angle on qubit 0, over weights from U(0, 2 pi).

    For ``strongly_entangling`` that angle is ``w[0, 0, 1]``: ``w[0, 0, 0]`` is
    an RZ acting on ``|0>`` and has zero gradient.

    Args:
        n_qubits: Circuit width.
        n_layers: Number of variational layers.
        ansatz: See `qedge.quantum.circuits.ANSATZE`.
        observable: ``"global"`` (Z on every qubit) or ``"local"`` (Z on qubit 0).
        samples: Number of random weight draws.
        seed: Seed for the draws.

    Raises:
        ValueError: For an unknown ``observable``.
    """
    if observable not in ("global", "local"):
        raise ValueError("observable must be 'global' or 'local'")
    wires = list(range(n_qubits))
    target = qml.prod(*(qml.PauliZ(w) for w in wires)) if observable == "global" else qml.PauliZ(0)

    @qml.qnode(qml.device("default.qubit", wires=n_qubits), diff_method="backprop")
    def cost(weights: Any) -> Any:
        variational_block(weights, wires, ansatz)
        return qml.expval(target)

    rng = np.random.default_rng(seed)
    shape = weight_shape(ansatz, n_layers, n_qubits)
    index = (0, 0, 1) if ansatz == "strongly_entangling" else (0, 0, 0)
    grads = []
    for _ in range(samples):
        weights = qml.numpy.array(rng.uniform(0, 2 * np.pi, shape), requires_grad=True)
        grads.append(float(qml.grad(cost)(weights)[index]))
    return float(np.var(grads))


def kernel_concentration(kernel: QuantumKernel, x: NDArray[Any]) -> dict[str, float]:
    """Mean and variance of the off-diagonal entries of ``kernel.matrix(x)``."""
    matrix = kernel.matrix(x)
    values = matrix[np.triu_indices(len(x), 1)]
    return {"mean": float(values.mean()), "variance": float(values.var())}


def haar_fidelity_histogram(n_qubits: int, bins: int) -> NDArray[np.float64]:
    """Probability of each fidelity bin on [0, 1] for Haar-random states.

    ``P(F) = (N - 1)(1 - F)^(N - 2)`` with ``N = 2**n_qubits``, integrated per bin.
    """
    dim = 2**n_qubits
    edges = np.linspace(0.0, 1.0, bins + 1)
    cumulative = 1 - (1 - edges) ** (dim - 1)
    return np.diff(cumulative)


def expressibility(
    state: Callable[[NDArray[Any]], NDArray[Any]],
    sample: Callable[[np.random.Generator], NDArray[Any]],
    n_qubits: int,
    *,
    pairs: int = 500,
    bins: int = 75,
    seed: int = 0,
) -> float:
    """KL divergence of sampled state fidelities from the Haar distribution.

    Args:
        state: Maps one sample (weights or inputs) to a state vector.
        sample: Draws one sample from a generator.
        n_qubits: Width of the states.
        pairs: Number of independent state pairs.
        bins: Histogram bins on [0, 1].
        seed: Seed for the samples.

    Returns:
        ``KL(P_circuit || P_Haar)``; lower means closer to Haar random.
    """
    rng = np.random.default_rng(seed)
    fidelities = []
    for _ in range(pairs):
        a, b = state(sample(rng)), state(sample(rng))
        fidelities.append(float(np.abs(np.vdot(a, b)) ** 2))
    counts, _ = np.histogram(fidelities, bins=bins, range=(0.0, 1.0))
    observed = counts / counts.sum()
    haar = haar_fidelity_histogram(n_qubits, bins)
    mask = observed > 0
    return float(np.sum(observed[mask] * np.log(observed[mask] / np.maximum(haar[mask], 1e-300))))


def ansatz_state(n_qubits: int, ansatz: str) -> Callable[[NDArray[Any]], NDArray[Any]]:
    """State vector after the variational block alone, as a function of the weights."""
    wires = list(range(n_qubits))

    @qml.qnode(qml.device("default.qubit", wires=n_qubits))
    def circuit(weights: Any) -> Any:
        variational_block(weights, wires, ansatz)
        return qml.state()

    return lambda weights: np.asarray(circuit(weights))


def ansatz_expressibility(
    n_qubits: int, n_layers: int, ansatz: str = "strongly_entangling", **kwargs: Any
) -> float:
    """`expressibility` of the ansatz over weights drawn from U(0, 2 pi)."""
    shape = weight_shape(ansatz, n_layers, n_qubits)
    return expressibility(
        ansatz_state(n_qubits, ansatz),
        lambda rng: rng.uniform(0, 2 * np.pi, shape),
        n_qubits,
        **kwargs,
    )


def scaling(values: Sequence[float]) -> float:
    """Slope of ``log(values)`` per step: about ``-log(2)`` per added qubit on a barren plateau."""
    logs = np.log(np.asarray(values, dtype=np.float64))
    return float(np.polyfit(np.arange(len(logs)), logs, 1)[0])
