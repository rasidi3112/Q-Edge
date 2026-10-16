"""Zero-noise extrapolation and measurement-error mitigation.

The circuits in this repository run on noiseless simulators, so these methods
have no noise to mitigate. With PennyLane 0.45,
`ZeroNoiseExtrapolation.execute_at_scale` returns NaN for scale factors above 1.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pennylane as qml
from numpy.typing import NDArray
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


class FoldingMethod(Enum):
    """How a circuit is folded to amplify noise.

    Giurgica-Tiron et al., "Digital zero noise extrapolation for quantum error
    mitigation", 2020.
    """

    GLOBAL = "global"  # Fold entire circuit: U → U·U†·U
    LOCAL = "local"  # Fold individual gates: G → G·G†·G


class ExtrapolationMethod(Enum):
    """Model fitted to (scale, value) pairs and evaluated at scale 0.

    ADAPTIVE scores each candidate on the full-data fit, not by cross-validation.
    """

    LINEAR = "linear"  # Linear extrapolation
    POLYNOMIAL = "polynomial"  # Polynomial fit
    EXPONENTIAL = "exponential"  # Exponential decay model
    RICHARDSON = "richardson"  # Richardson extrapolation
    ADAPTIVE = "adaptive"  # Adaptive method selection


@dataclass
class ZNEConfig:
    """Scale factors and fit settings.

    Raises:
        ValueError: If fewer than 2 scale factors are given, any is below 1, or
            they are not strictly increasing.
    """

    scale_factors: Sequence[float] = field(default_factory=lambda: [1.0, 1.5, 2.0, 2.5, 3.0])
    folding_method: FoldingMethod = FoldingMethod.GLOBAL
    extrapolation_method: ExtrapolationMethod = ExtrapolationMethod.RICHARDSON
    poly_degree: int = 2

    def __post_init__(self) -> None:
        if len(self.scale_factors) < 2:
            raise ValueError("At least 2 scale factors required for extrapolation")
        if min(self.scale_factors) < 1.0:
            raise ValueError("All scale factors must be >= 1.0")
        if not all(
            self.scale_factors[i] < self.scale_factors[i + 1]
            for i in range(len(self.scale_factors) - 1)
        ):
            raise ValueError("Scale factors must be strictly increasing")


class ZeroNoiseExtrapolation:
    """Zero-noise extrapolation.

    Temme et al., "Error mitigation for short-depth quantum circuits", 2017.

    Args:
        config: Scale factors and extrapolation method.
    """

    def __init__(self, config: ZNEConfig) -> None:
        self.config = config
        self.extrapolation_history: list[dict[str, Any]] = []

        logger.info(
            f"Initialized ZNE with scales={config.scale_factors}, "
            f"method={config.extrapolation_method.value}"
        )

    def fold_circuit_global(
        self,
        tape: qml.tape.QuantumTape,
        scale_factor: float,
    ) -> qml.tape.QuantumTape:
        """Return a tape implementing U (U^dagger U)^n plus a partial fold of the last gates.

        The partial fold is rounded to whole gates, so the achieved scale factor can
        differ from ``scale_factor``.
        """
        if scale_factor == 1.0:
            return tape

        # Calculate number of full folds and partial fold
        n_full_folds = int((scale_factor - 1) / 2)
        partial_fold = (scale_factor - 1) % 2 / 2

        # Get operations and observables
        operations = tape.operations.copy()
        measurements = tape.measurements

        # Apply full unitary folding: U → U(U†U)^n
        folded_ops = operations.copy()
        for _ in range(n_full_folds):
            # Add U†
            folded_ops.extend([qml.adjoint(op) for op in reversed(operations)])
            # Add U
            folded_ops.extend(operations.copy())

        # Handle partial folding
        if partial_fold > 0:
            n_gates_to_fold = int(len(operations) * partial_fold)
            if n_gates_to_fold > 0:
                gates_to_fold = operations[-n_gates_to_fold:]
                # Add partial U†
                folded_ops.extend([qml.adjoint(op) for op in reversed(gates_to_fold)])
                # Add partial U
                folded_ops.extend(gates_to_fold)

        # Create new tape
        with qml.tape.QuantumTape() as new_tape:
            for op in folded_ops:
                qml.apply(op)
            for m in measurements:
                qml.apply(m)

        return new_tape

    def fold_circuit_local(
        self,
        tape: qml.tape.QuantumTape,
        scale_factor: float,
    ) -> qml.tape.QuantumTape:
        """Return a tape where gates become G (G^dagger G)^k, assigned round-robin."""
        if scale_factor == 1.0:
            return tape

        operations = tape.operations.copy()
        measurements = tape.measurements

        # Calculate how many folds per gate
        n_gates = len(operations)
        target_ops = int(n_gates * scale_factor)
        ops_to_add = target_ops - n_gates

        # Distribute folds across gates
        folds_per_gate = np.zeros(n_gates, dtype=int)
        for i in range(ops_to_add // 2):
            gate_idx = i % n_gates
            folds_per_gate[gate_idx] += 1

        # Build folded circuit
        folded_ops = []
        for i, op in enumerate(operations):
            folded_ops.append(op)
            for _ in range(folds_per_gate[i]):
                folded_ops.append(qml.adjoint(op))
                folded_ops.append(op)

        # Create new tape
        with qml.tape.QuantumTape() as new_tape:
            for op in folded_ops:
                qml.apply(op)
            for m in measurements:
                qml.apply(m)

        return new_tape

    def execute_at_scale(
        self,
        circuit: Callable[..., Any],
        params: Any,
        scale_factor: float,
        device: qml.Device,
    ) -> float:
        """Fold ``circuit`` to ``scale_factor``, run it on ``device`` and return the mean."""
        if scale_factor == 1.0:
            # No folding needed
            return float(circuit(params))

        # Get the quantum tape
        tape = qml.tape.make_qscript(circuit)(params)

        # Apply folding
        if self.config.folding_method == FoldingMethod.GLOBAL:
            folded_tape = self.fold_circuit_global(tape, scale_factor)
        else:
            folded_tape = self.fold_circuit_local(tape, scale_factor)

        # Execute folded circuit
        results = device.execute(folded_tape)

        return float(np.mean(results))

    def _fit_linear(
        self,
        scales: NDArray[np.float64],
        values: NDArray[np.float64],
    ) -> float:
        coeffs = np.polyfit(scales, values, 1)
        return float(coeffs[1])  # y-intercept

    def _fit_polynomial(
        self,
        scales: NDArray[np.float64],
        values: NDArray[np.float64],
    ) -> float:
        degree = min(self.config.poly_degree, len(scales) - 1)
        coeffs = np.polyfit(scales, values, degree)
        return float(coeffs[-1])  # y-intercept (λ=0)

    def _fit_exponential(
        self,
        scales: NDArray[np.float64],
        values: NDArray[np.float64],
    ) -> float:
        def exp_model(x: NDArray[np.float64], a: float, b: float, c: float) -> NDArray[np.float64]:
            return a * np.exp(-b * x) + c

        try:
            # Initial guess
            p0 = [values[0] - values[-1], 1.0, values[-1]]
            popt, _ = curve_fit(exp_model, scales, values, p0=p0, maxfev=1000)
            return float(popt[0] + popt[2])  # Value at λ=0
        except (RuntimeError, ValueError):
            # Fall back to polynomial if exponential fails
            logger.warning("Exponential fit failed, falling back to polynomial")
            return self._fit_polynomial(scales, values)

    def _fit_richardson(
        self,
        scales: NDArray[np.float64],
        values: NDArray[np.float64],
    ) -> float:
        n = len(scales)

        # Build the Vandermonde-like matrix
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                A[i, j] = scales[i] ** j

        # Solve for coefficients
        try:
            # We want: sum(c_i * f(λ_i)) where c_i eliminates noise terms
            # This requires solving: A.T @ c = e_1 (first basis vector)
            e1 = np.zeros(n)
            e1[0] = 1.0
            c = np.linalg.solve(A.T, e1)

            return float(np.dot(c, values))
        except np.linalg.LinAlgError:
            logger.warning("Richardson extrapolation failed, falling back to linear")
            return self._fit_linear(scales, values)

    def _select_best_method(
        self,
        scales: NDArray[np.float64],
        values: NDArray[np.float64],
    ) -> tuple[float, str]:
        methods = {
            "linear": self._fit_linear,
            "polynomial": self._fit_polynomial,
            "exponential": self._fit_exponential,
            "richardson": self._fit_richardson,
        }

        best_method = "linear"
        best_value = methods["linear"](scales, values)
        best_cv_error = float("inf")

        for name, method in methods.items():
            try:
                # Leave-one-out cross-validation
                cv_errors = []
                for i in range(len(scales)):
                    train_scales = np.delete(scales, i)
                    train_values = np.delete(values, i)

                    # The held-out fit is unused: selection scores the full-data fit, so
                    # this is not leave-one-out cross-validation.
                    pred = method(train_scales, train_values)  # noqa: F841

                    # Estimate error as deviation from smooth curve
                    all_pred = method(scales, values)
                    cv_errors.append(abs(values[i] - all_pred))

                cv_error = np.mean(cv_errors)

                if cv_error < best_cv_error:
                    best_cv_error = cv_error
                    best_method = name
                    best_value = method(scales, values)

            except Exception as e:
                logger.debug(f"Method {name} failed: {e}")
                continue

        return best_value, best_method

    def extrapolate(
        self,
        scales: NDArray[np.float64],
        values: NDArray[np.float64],
    ) -> dict[str, Any]:
        """Fit ``values`` against ``scales`` and return the estimate at scale 0.

        If the exponential or Richardson fit fails, a polynomial or linear fit is used
        and ``method_used`` still names the requested method.

        Returns:
            ``mitigated_value``, ``method_used``, ``confidence`` and the raw inputs.
            ``confidence`` is 1 / (1 + std(residuals)) for linear and polynomial fits
            and a fixed 0.8 otherwise.
        """
        scales = np.array(scales)
        values = np.array(values)

        method = self.config.extrapolation_method

        if method == ExtrapolationMethod.LINEAR:
            mitigated = self._fit_linear(scales, values)
            method_name = "linear"
        elif method == ExtrapolationMethod.POLYNOMIAL:
            mitigated = self._fit_polynomial(scales, values)
            method_name = "polynomial"
        elif method == ExtrapolationMethod.EXPONENTIAL:
            mitigated = self._fit_exponential(scales, values)
            method_name = "exponential"
        elif method == ExtrapolationMethod.RICHARDSON:
            mitigated = self._fit_richardson(scales, values)
            method_name = "richardson"
        else:  # ADAPTIVE
            mitigated, method_name = self._select_best_method(scales, values)

        # Estimate confidence based on residuals
        confidence: float
        if method_name in ["linear", "polynomial"]:
            degree = 1 if method_name == "linear" else self.config.poly_degree
            coeffs = np.polyfit(scales, values, degree)
            predicted = np.polyval(coeffs, scales)
            residuals = values - predicted
            confidence = float(1.0 / (1.0 + np.std(residuals)))
        else:
            confidence = 0.8  # Default confidence for other methods

        result = {
            "mitigated_value": mitigated,
            "method_used": method_name,
            "confidence": confidence,
            "raw_scales": scales.tolist(),
            "raw_values": values.tolist(),
        }

        self.extrapolation_history.append(result)

        return result

    def mitigate(
        self,
        circuit: qml.QNode,
        params: Any,
        device: qml.Device | None = None,
    ) -> dict[str, Any]:
        """Execute at every scale factor and extrapolate.

        ``improvement`` is the distance between the mitigated value and the value at
        the largest scale factor, not the unmitigated value at scale 1.
        """
        if device is None:
            device = circuit.device

        # Execute at each noise scale
        noisy_values = []
        for scale in self.config.scale_factors:
            value = self.execute_at_scale(circuit, params, scale, device)
            noisy_values.append(value)

            logger.debug(f"Scale {scale:.2f}: value = {value:.6f}")

        # Extrapolate to zero noise
        result = self.extrapolate(
            np.array(self.config.scale_factors),
            np.array(noisy_values),
        )

        # Add comparison with unmitigated
        result["unmitigated_value"] = noisy_values[0]  # λ=1 value
        result["improvement"] = abs(result["mitigated_value"] - noisy_values[-1])

        logger.info(
            f"ZNE mitigation: {noisy_values[0]:.4f} → {result['mitigated_value']:.4f} "
            f"(method: {result['method_used']})"
        )

        return result


class MeasurementErrorMitigation:
    """Readout-error mitigation from a full 2^n x 2^n calibration matrix.

    Args:
        n_qubits: Number of measured qubits; calibration runs ``2**n_qubits``
            circuits.
        method: ``"inverse"``, ``"least_squares"`` (non-negative, sums to 1) or
            ``"iterative"``.

    Raises:
        ValueError: For an unknown ``method``.
    """

    def __init__(
        self,
        n_qubits: int,
        method: str = "inverse",
    ) -> None:
        if method not in ("inverse", "least_squares", "iterative"):
            raise ValueError(
                f"Unknown method {method!r}; expected 'inverse', 'least_squares' or 'iterative'"
            )
        self.n_qubits = n_qubits
        self.method = method

        self.calibration_matrix: NDArray[np.float64] | None = None
        self.inverse_matrix: NDArray[np.float64] | None = None
        self._is_calibrated = False

    def calibrate(self, device: qml.Device) -> None:
        """Prepare each basis state on ``device`` and store its measured probabilities."""
        n_states = 2**self.n_qubits
        self.calibration_matrix = np.zeros((n_states, n_states))

        @qml.qnode(device)
        def calibration_circuit(basis_state: int) -> Any:
            # Prepare |basis_state⟩ using X gates
            for qubit in range(self.n_qubits):
                if (basis_state >> qubit) & 1:
                    qml.PauliX(wires=qubit)
            return qml.probs(wires=range(self.n_qubits))

        for i in range(n_states):
            self.calibration_matrix[:, i] = calibration_circuit(i)

        # Compute inverse (or pseudo-inverse) for mitigation
        try:
            self.inverse_matrix = np.linalg.inv(self.calibration_matrix)
        except np.linalg.LinAlgError:
            self.inverse_matrix = np.linalg.pinv(self.calibration_matrix)

        self._is_calibrated = True

        fidelity = np.mean(np.diag(self.calibration_matrix))
        logger.info(f"Calibration complete. Readout fidelity: {fidelity:.4f}")

    def mitigate(self, measured_probs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Correct a measured probability vector of length ``2**n_qubits``.

        Raises:
            RuntimeError: If `calibrate` has not been called.
        """
        calibration, inverse = self.calibration_matrix, self.inverse_matrix
        if not self._is_calibrated or calibration is None or inverse is None:
            raise RuntimeError("Must call calibrate() before mitigating")

        if self.method == "inverse":
            mitigated = inverse @ measured_probs
        elif self.method == "least_squares":
            # Constrained least squares (non-negative, sums to 1)
            from scipy.optimize import minimize

            def objective(p: NDArray[np.float64]) -> float:
                return float(np.sum((calibration @ p - measured_probs) ** 2))

            constraints = [{"type": "eq", "fun": lambda p: np.sum(p) - 1.0}]
            bounds = [(0, 1) for _ in range(len(measured_probs))]

            result = minimize(
                objective,
                measured_probs,
                method="SLSQP",
                constraints=constraints,
                bounds=bounds,
            )
            mitigated = result.x
        else:
            # method == "iterative"
            mitigated = measured_probs.copy()
            for _ in range(10):
                correction = inverse @ (measured_probs - calibration @ mitigated)
                mitigated = mitigated + 0.5 * correction
                mitigated = np.clip(mitigated, 0, 1)
                mitigated = mitigated / np.sum(mitigated)

        # Ensure valid probability distribution
        mitigated = np.clip(mitigated, 0, 1)
        mitigated = mitigated / np.sum(mitigated)

        return mitigated

    def get_readout_fidelity(self) -> float:
        """Return the mean diagonal of the calibration matrix.

        Raises:
            RuntimeError: If `calibrate` has not been called.
        """
        if not self._is_calibrated:
            raise RuntimeError("Must call calibrate() first")

        return float(np.mean(np.diag(self.calibration_matrix)))
