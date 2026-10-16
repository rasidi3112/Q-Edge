from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np
import pennylane as qml
from numpy.typing import NDArray
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

class FeatureMapType(Enum):
    
    
    ZZ_FEATURE_MAP = "zz_feature_map"       # Diagonal + entanglement
    PAULI_FEATURE_MAP = "pauli_feature_map"  # Pauli rotations
    IQP_FEATURE_MAP = "iqp_feature_map"      # Instantaneous Quantum Polynomial
    AMPLITUDE_EMBEDDING = "amplitude"         # Amplitude encoding
    ANGLE_EMBEDDING = "angle"                 # Angle encoding

class KernelType(Enum):
    
    
    FIDELITY = "fidelity"        # Transition amplitude squared
    PROJECTED = "projected"       # Projected quantum kernel
    SWAP_TEST = "swap_test"       # SWAP test kernel

@dataclass
class QKAConfig:
    
    
    n_qubits: int = 8
    n_layers: int = 2
    feature_map_type: FeatureMapType = FeatureMapType.ZZ_FEATURE_MAP
    kernel_type: KernelType = KernelType.FIDELITY
    alignment_method: str = "L-BFGS-B"
    regularization: float = 0.01
    max_iterations: int = 100
    tolerance: float = 1e-6

class QuantumKernelAlignment:
    
    
    def __init__(
        self,
        config: QKAConfig,
        device_name: str = "default.qubit",
        seed: Optional[int] = None,
    ) -> None:
        
        self.config = config
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        
        # Create quantum device (needs 2x qubits for kernel evaluation)
        self.device = qml.device(
            device_name,
            wires=config.n_qubits * 2 if config.kernel_type == KernelType.SWAP_TEST 
            else config.n_qubits,
        )
        
        # Initialize trainable feature map parameters
        self._param_shape = self._get_param_shape()
        self.feature_map_params = self._initialize_params()
        
        # Build kernel circuit
        self._kernel_circuit = self._build_kernel_circuit()
        
        # Optimization history
        self.optimization_history: list[dict[str, Any]] = []
        
        logger.info(
            f"Initialized QKA with {config.n_qubits} qubits, "
            f"{config.feature_map_type.value} feature map"
        )
    
    def _get_param_shape(self) -> tuple[int, ...]:
        
        n_qubits = self.config.n_qubits
        n_layers = self.config.n_layers
        
        if self.config.feature_map_type == FeatureMapType.ZZ_FEATURE_MAP:
            # ZZ feature map: single parameter per qubit per layer + entanglement params
            return (n_layers, n_qubits + n_qubits * (n_qubits - 1) // 2)
        elif self.config.feature_map_type == FeatureMapType.PAULI_FEATURE_MAP:
            # Pauli feature map: 3 parameters per qubit per layer
            return (n_layers, n_qubits, 3)
        else:
            # Default: single parameter per qubit per layer
            return (n_layers, n_qubits)
    
    def _initialize_params(self) -> NDArray[np.float64]:
        
        # Small random initialization
        params = self._rng.uniform(-0.1, 0.1, self._param_shape)
        return params.astype(np.float64)
    
    def _build_kernel_circuit(self) -> qml.QNode:
        
        
        @qml.qnode(self.device, interface="autograd")
        def kernel_circuit(
            x1: NDArray[np.float64],
            x2: NDArray[np.float64],
            params: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            
            # Apply U(x2) - encode second data point
            self._apply_feature_map(x2, params, adjoint=False)
            
            # Apply U†(x1) - adjoint of first data point encoding
            self._apply_feature_map(x1, params, adjoint=True)
            
            # Return all probabilities - kernel value is |0...0⟩ probability
            return qml.probs(wires=range(self.config.n_qubits))
        
        return kernel_circuit
    
    def _apply_feature_map(
        self,
        x: NDArray[np.float64],
        params: NDArray[np.float64],
        adjoint: bool = False,
    ) -> None:
        
        n_qubits = self.config.n_qubits
        n_layers = self.config.n_layers
        
        # Ensure data fits qubits
        x_padded = np.zeros(n_qubits)
        x_padded[:min(len(x), n_qubits)] = x[:min(len(x), n_qubits)]
        
        layers = range(n_layers) if not adjoint else range(n_layers - 1, -1, -1)
        
        for layer in layers:
            if self.config.feature_map_type == FeatureMapType.ZZ_FEATURE_MAP:
                self._apply_zz_layer(x_padded, params[layer], adjoint)
            elif self.config.feature_map_type == FeatureMapType.PAULI_FEATURE_MAP:
                self._apply_pauli_layer(x_padded, params[layer], adjoint)
            else:
                self._apply_angle_layer(x_padded, params[layer], adjoint)
    
    def _apply_zz_layer(
        self,
        x: NDArray[np.float64],
        layer_params: NDArray[np.float64],
        adjoint: bool,
    ) -> None:
        
        n_qubits = self.config.n_qubits
        sign = -1.0 if adjoint else 1.0
        
        # Single qubit rotations
        for i in range(n_qubits):
            angle = sign * layer_params[i] * x[i]
            qml.RZ(angle, wires=i)
            qml.RY(angle, wires=i)
        
        # ZZ entanglement
        param_idx = n_qubits
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                angle = sign * layer_params[param_idx] * (np.pi - x[i]) * (np.pi - x[j])
                qml.CNOT(wires=[i, j])
                qml.RZ(angle, wires=j)
                qml.CNOT(wires=[i, j])
                param_idx += 1
    
    def _apply_pauli_layer(
        self,
        x: NDArray[np.float64],
        layer_params: NDArray[np.float64],
        adjoint: bool,
    ) -> None:
        
        n_qubits = self.config.n_qubits
        sign = -1.0 if adjoint else 1.0
        
        for i in range(n_qubits):
            qml.RX(sign * layer_params[i, 0] * x[i], wires=i)
            qml.RY(sign * layer_params[i, 1] * x[i], wires=i)
            qml.RZ(sign * layer_params[i, 2] * x[i], wires=i)
    
    def _apply_angle_layer(
        self,
        x: NDArray[np.float64],
        layer_params: NDArray[np.float64],
        adjoint: bool,
    ) -> None:
        
        n_qubits = self.config.n_qubits
        sign = -1.0 if adjoint else 1.0
        
        for i in range(n_qubits):
            qml.RY(sign * layer_params[i] * x[i], wires=i)
    
    def evaluate_kernel(
        self,
        x1: NDArray[np.float64],
        x2: NDArray[np.float64],
        params: Optional[NDArray[np.float64]] = None,
    ) -> float:
        
        if params is None:
            params = self.feature_map_params
        
        # Kernel circuit returns all probabilities, we need |0...0⟩ probability
        probs = self._kernel_circuit(x1, x2, params)
        return float(probs[0])
    
    def compute_kernel_matrix(
        self,
        X: NDArray[np.float64],
        Y: Optional[NDArray[np.float64]] = None,
        params: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        
        if params is None:
            params = self.feature_map_params
        
        if Y is None:
            Y = X
            symmetric = True
        else:
            symmetric = False
        
        n_x, n_y = len(X), len(Y)
        K = np.zeros((n_x, n_y))
        
        for i in range(n_x):
            start_j = i if symmetric else 0
            for j in range(start_j, n_y):
                K[i, j] = self.evaluate_kernel(X[i], Y[j], params)
                if symmetric and i != j:
                    K[j, i] = K[i, j]
        
        return K
    
    def kernel_alignment_score(
        self,
        K1: NDArray[np.float64],
        K2: NDArray[np.float64],
    ) -> float:
        
        # Frobenius inner product
        inner_product = np.sum(K1 * K2)
        
        # Frobenius norms
        norm_k1 = np.sqrt(np.sum(K1 * K1))
        norm_k2 = np.sqrt(np.sum(K2 * K2))
        
        if norm_k1 < 1e-10 or norm_k2 < 1e-10:
            return 0.0
        
        return inner_product / (norm_k1 * norm_k2)
    
    def align_to_target(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        kernel_type: str = "rbf",
    ) -> dict[str, Any]:
        
        # Compute target kernel matrix
        if kernel_type == "ideal":
            # Ideal kernel: K[i,j] = 1 if y[i] == y[j], else 0
            K_target = (y[:, None] == y[None, :]).astype(np.float64)
        elif kernel_type == "rbf":
            # RBF kernel with median heuristic for bandwidth
            from scipy.spatial.distance import pdist, squareform
            distances = squareform(pdist(X))
            sigma = np.median(distances[distances > 0])
            K_target = np.exp(-distances**2 / (2 * sigma**2))
        else:
            # Linear kernel
            K_target = X @ X.T
        
        # Center the target kernel
        K_target = self._center_kernel(K_target)
        
        # Optimization objective
        def objective(params_flat: NDArray[np.float64]) -> float:
            params = params_flat.reshape(self._param_shape)
            K_quantum = self.compute_kernel_matrix(X, params=params)
            K_quantum = self._center_kernel(K_quantum)
            
            # Negative alignment (minimize to maximize alignment)
            alignment = self.kernel_alignment_score(K_quantum, K_target)
            
            # L2 regularization
            reg = self.config.regularization * np.sum(params_flat**2)
            
            return -alignment + reg
        
        # Run optimization
        initial_params = self.feature_map_params.flatten()
        
        result = minimize(
            objective,
            initial_params,
            method=self.config.alignment_method,
            options={
                "maxiter": self.config.max_iterations,
                "ftol": self.config.tolerance,
            },
        )
        
        # Update parameters
        self.feature_map_params = result.x.reshape(self._param_shape)
        
        # Compute final alignment
        K_final = self.compute_kernel_matrix(X)
        final_alignment = self.kernel_alignment_score(
            self._center_kernel(K_final), K_target
        )
        
        optimization_result = {
            "success": result.success,
            "final_alignment": final_alignment,
            "n_iterations": result.nit,
            "final_objective": result.fun,
            "optimized_params": self.feature_map_params.copy(),
        }
        
        self.optimization_history.append(optimization_result)
        
        logger.info(
            f"Kernel alignment completed: alignment={final_alignment:.4f}, "
            f"iterations={result.nit}"
        )
        
        return optimization_result
    
    def _center_kernel(self, K: NDArray[np.float64]) -> NDArray[np.float64]:
        
        n = K.shape[0]
        one_n = np.ones((n, n)) / n
        
        return K - one_n @ K - K @ one_n + one_n @ K @ one_n
    
    def optimize_for_classification(
        self,
        X_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        X_val: Optional[NDArray[np.float64]] = None,
        y_val: Optional[NDArray[np.float64]] = None,
    ) -> dict[str, Any]:
        
        # First align to ideal kernel
        result = self.align_to_target(X_train, y_train, kernel_type="ideal")
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            K_val = self.compute_kernel_matrix(X_val, X_train)
            
            # Simple kernel-based classification (1-NN in kernel space)
            predictions = []
            for i in range(len(X_val)):
                nearest_idx = np.argmax(K_val[i])
                predictions.append(y_train[nearest_idx])
            
            predictions = np.array(predictions)
            accuracy = np.mean(predictions == y_val)
            result["validation_accuracy"] = accuracy
        
        return result
    
    def get_expressibility(
        self,
        n_samples: int = 1000,
    ) -> float:
        
        # Generate random input pairs
        fidelities = []
        
        for _ in range(n_samples):
            x1 = self._rng.uniform(-np.pi, np.pi, self.config.n_qubits)
            x2 = self._rng.uniform(-np.pi, np.pi, self.config.n_qubits)
            
            fidelity = self.evaluate_kernel(x1, x2)
            fidelities.append(fidelity)
        
        fidelities = np.array(fidelities)
        
        # Expressibility is measured by deviation from Haar distribution
        # For Haar random states, fidelity follows specific distribution
        hist, _ = np.histogram(fidelities, bins=50, range=(0, 1), density=True)
        
        # Haar distribution pdf for fidelity
        n = 2 ** self.config.n_qubits
        haar_hist = np.array([(n - 1) * (1 - f)**(n - 2) for f in np.linspace(0.01, 0.99, 50)])
        haar_hist = haar_hist / np.sum(haar_hist) * 50  # Normalize
        
        # KL divergence from Haar (lower is more expressible)
        hist = np.clip(hist, 1e-10, None)
        haar_hist = np.clip(haar_hist, 1e-10, None)
        kl_div = np.sum(hist * np.log(hist / haar_hist)) / 50
        
        # Convert to expressibility score (higher is better)
        expressibility = np.exp(-kl_div)
        
        return float(expressibility)
    
    def to_dict(self) -> dict[str, Any]:
        
        return {
            "config": {
                "n_qubits": self.config.n_qubits,
                "n_layers": self.config.n_layers,
                "feature_map_type": self.config.feature_map_type.value,
                "kernel_type": self.config.kernel_type.value,
                "regularization": self.config.regularization,
            },
            "feature_map_params": self.feature_map_params.tolist(),
            "optimization_history": self.optimization_history,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuantumKernelAlignment":
        
        config = QKAConfig(
            n_qubits=data["config"]["n_qubits"],
            n_layers=data["config"]["n_layers"],
            feature_map_type=FeatureMapType(data["config"]["feature_map_type"]),
            kernel_type=KernelType(data["config"]["kernel_type"]),
            regularization=data["config"]["regularization"],
        )
        
        qka = cls(config)
        qka.feature_map_params = np.array(data["feature_map_params"])
        qka.optimization_history = data.get("optimization_history", [])
        
        return qka
