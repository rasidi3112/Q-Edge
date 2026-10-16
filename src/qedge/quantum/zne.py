"""Zero-noise extrapolation and tensored readout-error mitigation.

ZNE (Temme et al., "Error mitigation for short-depth quantum circuits", 2017,
arXiv:1612.02058) runs a circuit at amplified noise and extrapolates the
results to zero noise. Noise is amplified by global unitary folding
U (U^dagger U)^n (Giurgica-Tiron et al., "Digital zero noise extrapolation
for quantum error mitigation", 2020, arXiv:2005.10921) with
`qml.noise.fold_global`; extrapolation uses PennyLane's Richardson,
polynomial and exponential fits; the exponential fit assumes the value
decays towards 0, as Pauli expectations do under depolarizing noise. It
only helps on a noisy device
(`qedge.quantum.noise.make_device`).

Folding works on whole gates, so the noise factor actually reached can
differ from the one requested; `achieved_scale_factors` reports it, and
`estimate` extrapolates over the achieved factors.
"""

from __future__ import annotations

import functools
import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pennylane as qml
from numpy.typing import NDArray
from pennylane.tape import QuantumScript
from scipy.optimize import minimize

METHODS = ("richardson", "poly", "exponential")


@dataclass(frozen=True)
class ZNEConfig:
    """Scale factors and extrapolation.

    Attributes:
        method: One of `METHODS`.
        scale_factors: Noise factors, starting at 1 and strictly increasing.
        order: Polynomial order for ``"poly"``.

    Raises:
        ValueError: If the method is unknown, the factors are invalid, or
            the polynomial order does not fit the number of factors.
    """

    method: str = "richardson"
    scale_factors: tuple[float, ...] = (1.0, 3.0, 5.0)
    order: int = 1

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"Unknown ZNE method {self.method!r}; expected {METHODS}")
        factors = self.scale_factors
        if (
            len(factors) < 2
            or factors[0] != 1
            or any(b <= a for a, b in itertools.pairwise(factors))
        ):
            raise ValueError("scale_factors must start at 1 and strictly increase")
        if self.method == "poly" and not 1 <= self.order < len(factors):
            raise ValueError("poly needs 1 <= order < number of scale factors")

    @property
    def odd_integer_factors(self) -> bool:
        """True when every factor is an odd integer, so folding reaches it exactly."""
        return all(float(f).is_integer() and int(f) % 2 == 1 for f in self.scale_factors)


def extrapolator(config: ZNEConfig) -> Callable[..., Any]:
    """The PennyLane extrapolation function for ``config``."""
    function: Callable[..., Any]
    if config.method == "richardson":
        function = qml.noise.richardson_extrapolate
    elif config.method == "poly":
        function = functools.partial(qml.noise.poly_extrapolate, order=config.order)
    else:
        function = qml.noise.exponential_extrapolate
    return function


def extrapolate(
    scale_factors: Sequence[float], values: Sequence[float], config: ZNEConfig
) -> float:
    """Extrapolate ``values`` measured at ``scale_factors`` to zero noise."""
    return float(extrapolator(config)(np.asarray(scale_factors), np.asarray(values)))


def mitigate(qnode: qml.QNode, config: ZNEConfig) -> Any:
    """Wrap ``qnode`` so that every call returns the extrapolated value.

    The wrapped QNode is differentiable like the original. PennyLane
    extrapolates over the requested factors, so only odd integer factors,
    which folding reaches exactly, are accepted here.

    Raises:
        ValueError: If a scale factor is not an odd integer.
    """
    if not config.odd_integer_factors:
        raise ValueError("mitigate needs odd integer scale factors; use estimate() otherwise")
    return qml.noise.mitigate_with_zne(
        qnode, list(config.scale_factors), qml.noise.fold_global, extrapolator(config)
    )


def achieved_scale_factors(tape: QuantumScript, scale_factors: Sequence[float]) -> list[float]:
    """Gate count of each folded tape divided by the gate count of ``tape``."""
    n_gates = len(tape.operations)
    achieved = []
    for factor in scale_factors:
        [folded], _ = qml.noise.fold_global(tape, factor)
        achieved.append(len(folded.operations) / n_gates)
    return achieved


@dataclass(frozen=True)
class ZNEResult:
    """Values at each noise factor and the extrapolated estimate.

    Attributes:
        requested: Scale factors from the config.
        achieved: Scale factors reached by folding.
        values: Expectation value at each achieved factor.
        mitigated: Extrapolation of ``values`` over ``achieved`` to 0.
    """

    requested: tuple[float, ...]
    achieved: tuple[float, ...]
    values: tuple[float, ...]
    mitigated: float

    @property
    def unmitigated(self) -> float:
        """The value at scale factor 1, i.e. the circuit as given."""
        return self.values[0]


def estimate(qnode: qml.QNode, args: Sequence[Any], config: ZNEConfig) -> ZNEResult:
    """Run ``qnode(*args)`` folded at every factor and extrapolate.

    ``qnode`` must return a single expectation value.
    """
    tape = qml.workflow.construct_tape(qnode)(*args)
    achieved = achieved_scale_factors(tape, config.scale_factors)
    values = []
    for factor in config.scale_factors:
        [folded], _ = qml.noise.fold_global(tape, factor)
        values.append(float(qml.execute([folded], qnode.device)[0]))
    return ZNEResult(
        requested=tuple(config.scale_factors),
        achieved=tuple(achieved),
        values=tuple(values),
        mitigated=extrapolate(achieved, values, config),
    )


class TensoredReadoutMitigation:
    """Readout-error mitigation from one 2x2 confusion matrix per qubit.

    Assumes readout errors are independent between qubits, so calibration
    needs ``2 * n_qubits`` circuits instead of ``2**n_qubits``.

    Args:
        n_qubits: Number of measured qubits.
        method: ``"inverse"`` (apply the inverse, then clip and renormalize)
            or ``"least_squares"`` (closest distribution under the model).

    Raises:
        ValueError: For an unknown ``method``.
    """

    def __init__(self, n_qubits: int, method: str = "least_squares") -> None:
        if method not in ("inverse", "least_squares"):
            raise ValueError(f"Unknown method {method!r}; expected 'inverse' or 'least_squares'")
        self.n_qubits = n_qubits
        self.method = method
        self.confusion: list[NDArray[np.float64]] = []
        self.calibration_circuits = 0

    def calibrate(self, device: Any) -> None:
        """Measure each qubit after preparing it in 0 and in 1 on ``device``.

        ``confusion[q][m, s]`` is the probability of reading ``m`` on qubit
        ``q`` when it was prepared in ``s``.
        """

        @qml.qnode(device)
        def prepare(qubit: int, bit: int) -> Any:
            if bit:
                qml.PauliX(wires=qubit)
            else:
                qml.Identity(wires=qubit)
            return qml.probs(wires=qubit)

        self.confusion = []
        self.calibration_circuits = 0
        for qubit in range(self.n_qubits):
            columns = []
            for bit in (0, 1):
                columns.append(np.asarray(prepare(qubit, bit), dtype=np.float64))
                self.calibration_circuits += 1
            self.confusion.append(np.stack(columns, axis=1))

    def _apply(self, matrices: Sequence[NDArray[np.float64]], probs: NDArray[Any]) -> NDArray[Any]:
        # Qubit 0 is the most significant bit in PennyLane's ordering.
        tensor = np.asarray(probs, dtype=np.float64).reshape([2] * self.n_qubits)
        for qubit, matrix in enumerate(matrices):
            tensor = np.moveaxis(np.tensordot(matrix, tensor, axes=([1], [qubit])), 0, qubit)
        return tensor.reshape(-1)

    def mitigate(self, measured: NDArray[Any]) -> NDArray[np.float64]:
        """Correct a measured distribution of length ``2**n_qubits``.

        Raises:
            RuntimeError: If `calibrate` has not been called.
        """
        if not self.confusion:
            raise RuntimeError("Must call calibrate() before mitigating")
        measured = np.asarray(measured, dtype=np.float64)
        if self.method == "inverse":
            corrected = self._apply([np.linalg.inv(m) for m in self.confusion], measured)
            corrected = np.clip(corrected, 0, None)
            return np.asarray(corrected / corrected.sum(), dtype=np.float64)

        def objective(p: NDArray[np.float64]) -> float:
            return float(np.sum((self._apply(self.confusion, p) - measured) ** 2))

        result = minimize(
            objective,
            measured,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * len(measured),
            constraints=[{"type": "eq", "fun": lambda p: np.sum(p) - 1.0}],
        )
        corrected = np.clip(result.x, 0, None)
        return np.asarray(corrected / corrected.sum(), dtype=np.float64)
