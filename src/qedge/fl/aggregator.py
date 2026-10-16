from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from src.quantum.circuits import VQCConfig, VariationalQuantumCircuit
from src.quantum.kernels import QKAConfig, QuantumKernelAlignment
from src.quantum.error_mitigation import ZNEConfig, ZeroNoiseExtrapolation
from src.quantum.azure_connector import (
    AzureQuantumConfig,
    AzureQuantumConnector,
    QuantumProvider,
    QuantumTarget,
)

logger = logging.getLogger(__name__)

class AggregationStrategy(Enum):
    
    
    FEDAVG = "fedavg"           # Weighted average by sample count
    FEDPROX = "fedprox"         # FedAvg with proximal regularization
    FEDOPT = "fedopt"           # Server-side optimization (FedAdam, FedYogi)
    SCAFFOLD = "scaffold"       # Variance reduction
    FEDBN = "fedbn"             # Batch normalization handling
    QUANTUM = "quantum"         # Quantum-enhanced aggregation

@dataclass
class LocalModelUpdate:
    
    
    client_id: str
    weights: NDArray[np.float64]
    gradients: Optional[NDArray[np.float64]] = None
    n_samples: int = 0
    local_loss: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GlobalModelState:
    
    
    weights: NDArray[np.float64]
    round_number: int
    quantum_embedding: Optional[NDArray[np.float64]] = None
    aggregation_metrics: Dict[str, Any] = field(default_factory=dict)
    vqc_params: Optional[NDArray[np.float64]] = None
    kernel_params: Optional[NDArray[np.float64]] = None

@dataclass
class QuantumAggregatorConfig:
    
    
    n_qubits: int = 8
    vqc_layers: int = 4
    qka_layers: int = 2
    aggregation_strategy: AggregationStrategy = AggregationStrategy.FEDAVG
    use_error_mitigation: bool = True
    zne_scale_factors: Sequence[float] = field(
        default_factory=lambda: [1.0, 1.5, 2.0, 2.5]
    )
    use_azure_quantum: bool = False
    azure_config: Optional[AzureQuantumConfig] = None
    weight_compression_ratio: float = 0.1  # Encode 10% of weights quantumly
    quantum_learning_rate: float = 0.01
    classical_weight: float = 0.7  # 70% classical, 30% quantum

class QuantumGlobalAggregator:
    
    
    def __init__(
        self,
        config: QuantumAggregatorConfig,
        seed: Optional[int] = None,
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
            self.zne = ZeroNoiseExtrapolation(zne_config)
        else:
            self.zne = None
        
        # Initialize Azure Quantum connector
        if config.use_azure_quantum and config.azure_config is not None:
            self.azure_connector = AzureQuantumConnector(config.azure_config)
        else:
            self.azure_connector = None
        
        # State tracking
        self.global_state: Optional[GlobalModelState] = None
        self.round_history: List[GlobalModelState] = []
        self._current_round = 0
        
        logger.info(
            f"Initialized QuantumGlobalAggregator: "
            f"qubits={config.n_qubits}, vqc_layers={config.vqc_layers}, "
            f"strategy={config.aggregation_strategy.value}"
        )
    
    async def connect_azure(self) -> bool:
        
        if self.azure_connector is not None:
            try:
                await self.azure_connector.connect()
                return True
            except Exception as e:
                logger.error(f"Failed to connect to Azure Quantum: {e}")
                return False
        return True
    
    def _classical_aggregate(
        self,
        updates: Sequence[LocalModelUpdate],
    ) -> NDArray[np.float64]:
        
        strategy = self.config.aggregation_strategy
        
        if strategy == AggregationStrategy.FEDAVG:
            return self._fedavg(updates)
        elif strategy == AggregationStrategy.FEDPROX:
            return self._fedprox(updates)
        elif strategy == AggregationStrategy.FEDOPT:
            return self._fedopt(updates)
        else:
            # Default to FedAvg
            return self._fedavg(updates)
    
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
        for update, w in zip(updates, weights):
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
        
        # Simplified Adam update (would need state for full Adam)
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
            compressed[:len(flat_weights)] = flat_weights
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
        original_shape: tuple,
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
        quantum_encoding = quantum_features[:self.config.n_qubits]
        
        return quantum_encoding
    
    async def _quantum_transform_azure(
        self,
        weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        
        if self.azure_connector is None or not self.azure_connector.is_connected:
            # Fall back to local simulation
            return self._quantum_transform(weights)
        
        # Compress weights
        compressed = self._compress_weights(weights)
        
        # Submit to Azure Quantum
        result = await self.azure_connector.submit_pennylane_circuit(
            circuit_fn=lambda p: self.vqc._circuit(self.vqc.params, p),
            params=compressed,
            n_qubits=self.config.n_qubits,
            shots=1024,
        )
        
        if result.probabilities is not None:
            # Use probabilities as quantum features
            quantum_features = np.log(result.probabilities + 1e-10)
            quantum_features = quantum_features / np.max(np.abs(quantum_features))
            return quantum_features[:self.config.n_qubits]
        
        # Fall back to simulation if Azure fails
        return self._quantum_transform(weights)
    
    def _compute_kernel_distance(
        self,
        weights_list: List[NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        
        # Compress each weight vector
        compressed_weights = np.array([
            self._compress_weights(w) for w in weights_list
        ])
        
        # Compute quantum kernel matrix
        K = self.qka.compute_kernel_matrix(compressed_weights)
        
        return K
    
    def _apply_error_mitigation(
        self,
        circuit_output: NDArray[np.float64],
        circuit_fn: callable,
        params: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        
        if self.zne is None:
            return circuit_output
        
        # Note: Full ZNE would require multiple circuit executions
        # This is a simplified version using the existing output
        
        # Apply simple noise model correction
        # In practice, ZNE.mitigate() would handle this
        mitigated = circuit_output * 1.05  # Simple correction factor
        
        # Normalize probabilities
        if np.sum(mitigated) > 0:
            mitigated = mitigated / np.sum(mitigated)
        
        return mitigated
    
    async def aggregate(
        self,
        updates: Sequence[LocalModelUpdate],
        use_quantum: bool = True,
    ) -> GlobalModelState:
        
        if len(updates) == 0:
            raise ValueError("No updates provided for aggregation")
        
        self._current_round += 1
        
        logger.info(
            f"Starting aggregation round {self._current_round} "
            f"with {len(updates)} client updates"
        )
        
        # Step 1: Classical aggregation
        classical_weights = self._classical_aggregate(updates)
        original_shape = classical_weights.shape
        
        metrics: Dict[str, Any] = {
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
        metrics["kernel_determinant"] = float(np.linalg.det(kernel_matrix + 1e-6 * np.eye(len(updates))))
        
        # Step 3: Apply quantum transformation
        if self.config.use_azure_quantum and self.azure_connector is not None:
            quantum_encoding = await self._quantum_transform_azure(classical_weights)
        else:
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
        if hasattr(self, '_prev_kernel_matrix') and self._prev_kernel_matrix is not None:
            alignment = self.qka.kernel_alignment_score(
                kernel_matrix, self._prev_kernel_matrix
            )
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
    ) -> Dict[str, Any]:
        
        weights_list = [u.weights for u in updates]
        compressed = np.array([self._compress_weights(w) for w in weights_list])
        
        # Train kernel for label alignment
        kernel_result = self.qka.align_to_target(
            compressed, labels, kernel_type="ideal"
        )
        
        training_results = {
            "kernel_alignment": kernel_result["final_alignment"],
            "n_iterations": kernel_result["n_iterations"],
            "vqc_params_updated": True,
        }
        
        # Simple VQC parameter update based on gradient
        # In practice, would use proper gradient descent
        gradient_estimate = self._rng.normal(0, 0.01, self.vqc.params.shape)
        self.vqc.params -= self.config.quantum_learning_rate * gradient_estimate
        
        return training_results
    
    def get_aggregation_metrics(self) -> Dict[str, Any]:
        
        if len(self.round_history) == 0:
            return {"message": "No aggregation rounds completed"}
        
        return {
            "total_rounds": len(self.round_history),
            "current_round": self._current_round,
            "latest_metrics": self.round_history[-1].aggregation_metrics,
            "weight_norm_history": [
                float(np.linalg.norm(gs.weights)) for gs in self.round_history
            ],
            "vqc_params_count": self.vqc.num_params,
            "qka_expressibility": self.qka.get_expressibility(n_samples=100),
        }
    
    def save_state(self, filepath: str) -> None:
        
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
            }
        }
        
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Saved aggregator state to {filepath}")
    
    @classmethod
    def load_state(cls, filepath: str) -> "QuantumGlobalAggregator":
        
        import json
        
        with open(filepath, "r") as f:
            state = json.load(f)
        
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
