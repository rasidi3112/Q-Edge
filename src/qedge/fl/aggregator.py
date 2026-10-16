"""Server-side aggregation of client weight vectors with an optional circuit step.

With ``use_quantum=True``, `QuantumGlobalAggregator.aggregate` replaces
``1 - classical_weight`` of every weight (30% by default) with values in [-1, 0]
derived from the log-probabilities of a variational circuit evaluated on
``n_qubits`` summary statistics of the aggregate. FEDPROX and FEDOPT are not the
published algorithms: FEDPROX mixes the new aggregate with the previous one on
the server, and FEDOPT takes a fixed step of 0.01 toward the FedAvg result.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from qedge.quantum.circuits import VariationalQuantumCircuit, VQCConfig
from qedge.quantum.kernels import QKAConfig, QuantumKernelAlignment
from qedge.quantum.zne import ZeroNoiseExtrapolation, ZNEConfig

logger = logging.getLogger(__name__)


class AggregationStrategy(Enum):
    """Server-side rule for combining client weights (see the module docstring)."""

    FEDAVG = "fedavg"  # Weighted average by sample count
    FEDPROX = "fedprox"  # FedAvg with proximal regularization
    FEDOPT = "fedopt"  # Server-side optimization (FedAdam, FedYogi)


@dataclass
class LocalModelUpdate:
    """One client's flattened weights for a round, with its claimed sample count and loss."""

    client_id: str
    weights: NDArray[np.float64]
    gradients: NDArray[np.float64] | None = None
    n_samples: int = 0
    local_loss: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GlobalModelState:
    """Result of one aggregation round."""

    weights: NDArray[np.float64]
    round_number: int
    quantum_embedding: NDArray[np.float64] | None = None
    aggregation_metrics: dict[str, Any] = field(default_factory=dict)
    vqc_params: NDArray[np.float64] | None = None
    kernel_params: NDArray[np.float64] | None = None


@dataclass
class QuantumAggregatorConfig:
    """Configuration for `QuantumGlobalAggregator`.

    ``classical_weight`` is the share of the classical aggregate kept when
    ``aggregate`` runs with ``use_quantum=True``.
    """

    n_qubits: int = 8
    vqc_layers: int = 4
    qka_layers: int = 2
    aggregation_strategy: AggregationStrategy = AggregationStrategy.FEDAVG
    use_error_mitigation: bool = True
    zne_scale_factors: Sequence[float] = field(default_factory=lambda: [1.0, 1.5, 2.0, 2.5])
    quantum_learning_rate: float = 0.01
    classical_weight: float = 0.7  # 70% classical, 30% quantum


class QuantumGlobalAggregator:
    """Aggregates `LocalModelUpdate` objects round by round.

    Args:
        config: Aggregation and circuit settings.
        seed: Seed for circuit and kernel parameter initialization.
    """

    def __init__(
        self,
        config: QuantumAggregatorConfig,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # Initialize VQC
        vqc_config = VQCConfig(
            n_qubits=config.n_qubits,
            n_layers=config.vqc_layers,
            data_reuploading=True,
        )
        self.vqc = VariationalQuantumCircuit(vqc_config, seed=seed)

        # Initialize QKA
        qka_config = QKAConfig(
            n_qubits=config.n_qubits,
            n_layers=config.qka_layers,
        )
        self.qka = QuantumKernelAlignment(qka_config, seed=seed)

        # Initialize ZNE
        if config.use_error_mitigation:
            zne_config = ZNEConfig(
                scale_factors=list(config.zne_scale_factors),
            )
            self.zne: ZeroNoiseExtrapolation | None = ZeroNoiseExtrapolation(zne_config)
        else:
            self.zne = None

        # State tracking
        self.global_state: GlobalModelState | None = None
        self.round_history: list[GlobalModelState] = []
        self._current_round = 0
        self._prev_kernel_matrix: NDArray[np.float64] | None = None

        logger.info(
            f"Initialized QuantumGlobalAggregator: "
            f"qubits={config.n_qubits}, vqc_layers={config.vqc_layers}, "
            f"strategy={config.aggregation_strategy.value}"
        )

    def _classical_aggregate(
        self,
        updates: Sequence[LocalModelUpdate],
    ) -> NDArray[np.float64]:
        strategy = self.config.aggregation_strategy

        if strategy == AggregationStrategy.FEDAVG:
            return self._fedavg(updates)
        if strategy == AggregationStrategy.FEDPROX:
            return self._fedprox(updates)
        if strategy == AggregationStrategy.FEDOPT:
            return self._fedopt(updates)
        raise ValueError(f"Unsupported aggregation strategy: {strategy}")

    def _fedavg(
        self,
        updates: Sequence[LocalModelUpdate],
    ) -> NDArray[np.float64]:
        total_samples = sum(u.n_samples for u in updates)

        if total_samples == 0:
            # Equal weighting if no sample counts
            weights = [1.0 / len(updates)] * len(updates)
        else:
            weights = [u.n_samples / total_samples for u in updates]

        # Ensure all weight arrays have the same shape
        weight_shape = updates[0].weights.shape

        aggregated = np.zeros(weight_shape)
        for update, w in zip(updates, weights, strict=True):
            aggregated += w * update.weights

        return aggregated

    def _fedprox(
        self,
        updates: Sequence[LocalModelUpdate],
        mu: float = 0.01,
    ) -> NDArray[np.float64]:
        # First apply FedAvg
        aggregated = self._fedavg(updates)

        # Apply proximal regularization towards previous global
        if self.global_state is not None:
            prev_global = self.global_state.weights
            aggregated = (1 - mu) * aggregated + mu * prev_global

        return aggregated

    def _fedopt(
        self,
        updates: Sequence[LocalModelUpdate],
    ) -> NDArray[np.float64]:
        # Compute pseudo-gradient
        aggregated_update = self._fedavg(updates)

        if self.global_state is None:
            return aggregated_update

        prev_global = self.global_state.weights
        pseudo_gradient = aggregated_update - prev_global

        # Fixed-step move toward the FedAvg result; no optimizer state, not Adam.
        learning_rate = 0.01
        updated = prev_global + learning_rate * pseudo_gradient

        return updated

    def _compress_weights(
        self,
        weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        flat_weights = weights.flatten()

        # Target dimension is n_qubits
        target_dim = self.config.n_qubits

        if len(flat_weights) <= target_dim:
            # Pad if smaller
            compressed = np.zeros(target_dim)
            compressed[: len(flat_weights)] = flat_weights
        else:
            # Use strided sampling with weighted average
            stride = len(flat_weights) // target_dim
            compressed = np.zeros(target_dim)
            for i in range(target_dim):
                start = i * stride
                end = min(start + stride, len(flat_weights))
                compressed[i] = np.mean(flat_weights[start:end])

        # Normalize to [-π, π] for angle encoding
        if np.std(compressed) > 0:
            compressed = (compressed - np.mean(compressed)) / np.std(compressed)
        compressed = np.tanh(compressed) * np.pi

        return compressed

    def _expand_weights(
        self,
        compressed: NDArray[np.float64],
        original_shape: tuple[int, ...],
    ) -> NDArray[np.float64]:
        n_elements = np.prod(original_shape)
        target_dim = len(compressed)

        # Expand using interpolation
        expanded = np.zeros(n_elements)
        stride = n_elements // target_dim

        for i in range(target_dim):
            start = i * stride
            end = min(start + stride, n_elements)
            expanded[start:end] = compressed[i]

        return expanded.reshape(original_shape)

    def _quantum_transform(
        self,
        weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        # Compress weights for quantum encoding
        compressed = self._compress_weights(weights)

        # Forward pass through VQC
        output_probs = self.vqc.forward(compressed)

        # Extract quantum features from probabilities
        # Use log-amplitudes as features
        quantum_features = np.log(output_probs + 1e-10)

        # Normalize
        quantum_features = quantum_features / np.max(np.abs(quantum_features))

        # Take first n_qubits features
        quantum_encoding = quantum_features[: self.config.n_qubits]

        return cast(NDArray[np.float64], quantum_encoding)

    def _compute_kernel_distance(
        self,
        weights_list: list[NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        # Compress each weight vector
        compressed_weights = np.array([self._compress_weights(w) for w in weights_list])

        # Compute quantum kernel matrix
        K = self.qka.compute_kernel_matrix(compressed_weights)

        return K

    def _apply_error_mitigation(
        self,
        circuit_output: NDArray[np.float64],
        circuit_fn: Callable[..., Any],
        params: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if self.zne is None:
            return circuit_output

        # Not error mitigation: scaling by 1.05 and renormalizing returns the input
        # distribution unchanged. Nothing calls this method.
        mitigated = circuit_output * 1.05

        # Normalize probabilities
        if np.sum(mitigated) > 0:
            mitigated = mitigated / np.sum(mitigated)

        return mitigated

    async def aggregate(
        self,
        updates: Sequence[LocalModelUpdate],
        use_quantum: bool = True,
    ) -> GlobalModelState:
        """Run one aggregation round.

        Args:
            updates: Client updates. Weight arrays must share one shape; this is not
                checked.
            use_quantum: Mix in the circuit-derived values and compute an
                ``n_clients x n_clients`` quantum kernel for the metrics. The kernel
                costs O(n_clients^2) circuit evaluations and does not change the
                weights.

        Returns:
            The new global state, also stored on ``self.global_state``.

        Raises:
            ValueError: If ``updates`` is empty.
        """
        if len(updates) == 0:
            raise ValueError("No updates provided for aggregation")

        self._current_round += 1

        logger.info(
            f"Starting aggregation round {self._current_round} with {len(updates)} client updates"
        )

        # Step 1: Classical aggregation
        classical_weights = self._classical_aggregate(updates)
        original_shape = classical_weights.shape

        metrics: dict[str, Any] = {
            "round": self._current_round,
            "n_clients": len(updates),
            "total_samples": sum(u.n_samples for u in updates),
            "avg_local_loss": np.mean([u.local_loss for u in updates]),
        }

        if not use_quantum:
            # Pure classical aggregation
            self.global_state = GlobalModelState(
                weights=classical_weights,
                round_number=self._current_round,
                aggregation_metrics=metrics,
            )
            self.round_history.append(self.global_state)
            return self.global_state

        # Step 2: Compute quantum kernel over client updates
        weights_list = [u.weights for u in updates]
        kernel_matrix = self._compute_kernel_distance(weights_list)
        metrics["kernel_determinant"] = float(
            np.linalg.det(kernel_matrix + 1e-6 * np.eye(len(updates)))
        )

        # Step 3: Apply quantum transformation
        quantum_encoding = self._quantum_transform(classical_weights)

        # Step 4: Expand quantum encoding to weight space
        quantum_weights = self._expand_weights(quantum_encoding, original_shape)

        # Step 5: Hybrid combination
        alpha = self.config.classical_weight
        hybrid_weights = alpha * classical_weights + (1 - alpha) * quantum_weights

        metrics["classical_weight"] = alpha
        metrics["quantum_norm"] = float(np.linalg.norm(quantum_weights))
        metrics["hybrid_norm"] = float(np.linalg.norm(hybrid_weights))

        # Step 6: Compute kernel alignment quality
        if self._prev_kernel_matrix is not None:
            alignment = self.qka.kernel_alignment_score(kernel_matrix, self._prev_kernel_matrix)
            metrics["kernel_alignment"] = float(alignment)
        self._prev_kernel_matrix = kernel_matrix

        # Create global state
        self.global_state = GlobalModelState(
            weights=hybrid_weights,
            round_number=self._current_round,
            quantum_embedding=quantum_encoding,
            aggregation_metrics=metrics,
            vqc_params=self.vqc.params.copy(),
            kernel_params=self.qka.feature_map_params.copy(),
        )

        self.round_history.append(self.global_state)

        logger.info(
            f"Aggregation round {self._current_round} complete. "
            f"Hybrid weight norm: {metrics['hybrid_norm']:.4f}"
        )

        return self.global_state

    async def train_quantum_parameters(
        self,
        updates: Sequence[LocalModelUpdate],
        labels: NDArray[np.float64],
        n_epochs: int = 10,
    ) -> dict[str, Any]:
        """Align the kernel to ``labels`` and perturb the circuit parameters.

        The circuit update is Gaussian noise (sigma 0.01) scaled by
        ``quantum_learning_rate``, not a gradient step.
        """
        weights_list = [u.weights for u in updates]
        compressed = np.array([self._compress_weights(w) for w in weights_list])

        # Train kernel for label alignment
        kernel_result = self.qka.align_to_target(compressed, labels, kernel_type="ideal")

        training_results = {
            "kernel_alignment": kernel_result["final_alignment"],
            "n_iterations": kernel_result["n_iterations"],
            "vqc_params_updated": True,
        }

        # Random perturbation, not a gradient step.
        gradient_estimate = self._rng.normal(0, 0.01, self.vqc.params.shape)
        self.vqc.params -= self.config.quantum_learning_rate * gradient_estimate

        return training_results

    def get_aggregation_metrics(self) -> dict[str, Any]:
        """Summarize completed rounds.

        Once a round has completed, every call evaluates 100 kernel circuits for the
        expressibility estimate.
        """
        if len(self.round_history) == 0:
            return {"message": "No aggregation rounds completed"}

        return {
            "total_rounds": len(self.round_history),
            "current_round": self._current_round,
            "latest_metrics": self.round_history[-1].aggregation_metrics,
            "weight_norm_history": [float(np.linalg.norm(gs.weights)) for gs in self.round_history],
            "vqc_params_count": self.vqc.num_params,
            "qka_expressibility": self.qka.get_expressibility(n_samples=100),
        }

    def save_state(self, filepath: str) -> None:
        """Write config, circuit parameters and the latest global weights to JSON."""
        import json

        state = {
            "config": {
                "n_qubits": self.config.n_qubits,
                "vqc_layers": self.config.vqc_layers,
                "qka_layers": self.config.qka_layers,
                "aggregation_strategy": self.config.aggregation_strategy.value,
            },
            "vqc": self.vqc.to_dict(),
            "qka": self.qka.to_dict(),
            "current_round": self._current_round,
            "global_state": {
                "weights": self.global_state.weights.tolist() if self.global_state else None,
                "round_number": self.global_state.round_number if self.global_state else 0,
            },
        }

        Path(filepath).write_text(json.dumps(state, indent=2))

        logger.info(f"Saved aggregator state to {filepath}")

    @classmethod
    def load_state(cls, filepath: str) -> QuantumGlobalAggregator:
        """Rebuild an aggregator from `save_state` output; round history is not restored."""
        import json

        state = json.loads(Path(filepath).read_text())

        config = QuantumAggregatorConfig(
            n_qubits=state["config"]["n_qubits"],
            vqc_layers=state["config"]["vqc_layers"],
            qka_layers=state["config"]["qka_layers"],
            aggregation_strategy=AggregationStrategy(state["config"]["aggregation_strategy"]),
        )

        aggregator = cls(config)
        aggregator.vqc = VariationalQuantumCircuit.from_dict(state["vqc"])
        aggregator.qka = QuantumKernelAlignment.from_dict(state["qka"])
        aggregator._current_round = state["current_round"]

        if state["global_state"]["weights"] is not None:
            aggregator.global_state = GlobalModelState(
                weights=np.array(state["global_state"]["weights"]),
                round_number=state["global_state"]["round_number"],
            )

        logger.info(f"Loaded aggregator state from {filepath}")
        return aggregator
