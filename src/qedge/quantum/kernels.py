"""Fidelity quantum kernels.

``k(x1, x2) = |<0| U(x1)^dagger U(x2) |0>|^2``, evaluated as the probability
of the all-zeros outcome after ``U(x2)`` followed by ``qml.adjoint(U)(x1)``.
The adjoint reverses the gate order as well as the angles, so ``k(x, x) = 1``
and the Gram matrix is positive semidefinite for any parameters.

Feature maps:

- ``"zz"``: per layer, Hadamards, ``RZ(w_i x_i)`` on each qubit and
  ``exp(-i w_ij (pi - x_i)(pi - x_j) Z_i Z_j)`` on neighbouring pairs
  (Havlíček et al., "Supervised learning with quantum-enhanced feature
  spaces", Nature 2019).
- ``"angle"``: per layer, ``RY(w_i x_i)`` then a CNOT chain.

``w`` are bandwidth weights, 1 by default. With 4 standardized features the
``"zz"`` map at bandwidth 1 already gives near-zero off-diagonal values, so
``"angle"`` is the default. Kernel values concentrate around
a constant as qubits are added (Thanasilp et al., "Exponential
concentration in quantum kernel methods", 2022, arXiv:2208.11060); see
`qedge.quantum.diagnostics.kernel_concentration`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pennylane as qml
from numpy.typing import NDArray

FEATURE_MAPS = ("angle", "zz")


def parameter_shape(feature_map: str, n_layers: int, n_qubits: int) -> tuple[int, int]:
    """Shape of the bandwidth weights.

    Raises:
        ValueError: If ``feature_map`` is unknown.
    """
    if feature_map == "zz":
        return (n_layers, n_qubits + max(n_qubits - 1, 0))
    if feature_map == "angle":
        return (n_layers, n_qubits)
    raise ValueError(f"Unknown feature map {feature_map!r}; expected {FEATURE_MAPS}")


def feature_map(x: Any, params: Any, wires: Sequence[int], kind: str) -> None:
    """Apply ``U(x)`` for the feature map ``kind``."""
    n = len(wires)
    for layer in range(params.shape[0]):
        w = params[layer]
        if kind == "zz":
            for i, wire in enumerate(wires):
                qml.Hadamard(wires=wire)
                qml.RZ(w[i] * x[i], wires=wire)
            for i in range(n - 1):
                angle = w[n + i] * (np.pi - x[i]) * (np.pi - x[i + 1])
                qml.IsingZZ(angle, wires=[wires[i], wires[i + 1]])
        else:
            for i, wire in enumerate(wires):
                qml.RY(w[i] * x[i], wires=wire)
            for i in range(n - 1):
                qml.CNOT(wires=[wires[i], wires[i + 1]])


class QuantumKernel:
    """Fidelity kernel on ``default.qubit``.

    Args:
        n_qubits: One qubit per input feature.
        n_layers: Repetitions of the feature map.
        kind: One of `FEATURE_MAPS`.
        params: Bandwidth weights of `parameter_shape`; all ones if None.

    Raises:
        ValueError: If ``kind`` is unknown or ``params`` has the wrong shape.
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 1,
        kind: str = "angle",
        params: NDArray[np.float64] | None = None,
    ) -> None:
        shape = parameter_shape(kind, n_layers, n_qubits)
        self.params = np.ones(shape) if params is None else np.asarray(params, dtype=np.float64)
        if self.params.shape != shape:
            raise ValueError(f"params has shape {self.params.shape}, expected {shape}")
        self.n_qubits = n_qubits
        self.kind = kind
        wires = list(range(n_qubits))

        @qml.qnode(qml.device("default.qubit", wires=n_qubits))
        def circuit(x1: Any, x2: Any) -> Any:
            feature_map(x2, self.params, wires, kind)
            qml.adjoint(feature_map)(x1, self.params, wires, kind)
            return qml.probs(wires=wires)

        self._circuit = circuit

    def __call__(self, x1: NDArray[Any], x2: NDArray[Any]) -> float:
        """Kernel value in [0, 1].

        Raises:
            ValueError: If an input does not have one value per qubit.
        """
        if len(x1) != self.n_qubits or len(x2) != self.n_qubits:
            raise ValueError(f"inputs must have {self.n_qubits} features")
        return float(self._circuit(x1, x2)[0])

    def matrix(self, x: NDArray[Any], y: NDArray[Any] | None = None) -> NDArray[np.float64]:
        """``K[i, j] = k(x[i], y[j])``; with ``y=None`` the symmetric Gram matrix of ``x``."""
        if y is None:
            return np.asarray(
                qml.kernels.square_kernel_matrix(x, self, assume_normalized_kernel=True)
            )
        return np.asarray(qml.kernels.kernel_matrix(x, y, self))


def target_alignment(matrix: NDArray[Any], labels: NDArray[Any]) -> float:
    """Kernel-target alignment ``<K, yy^T>_F / (||K||_F ||yy^T||_F)`` with labels in {-1, 1}.

    Cristianini et al., "On Kernel-Target Alignment", NIPS 2001.
    """
    target = np.outer(labels, labels)
    return float(np.sum(matrix * target) / (np.linalg.norm(matrix) * np.linalg.norm(target)))
