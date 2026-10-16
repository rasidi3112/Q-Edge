"""Variational quantum circuits on PennyLane simulators."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pennylane as qml
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class EntanglementPattern(Enum):
    """CNOT layout applied after each rotation layer."""

    LINEAR = "linear"  # Nearest-neighbor entanglement
    CIRCULAR = "circular"  # Ring topology
    FULL = "full"  # All-to-all connectivity
    STAR = "star"  # Hub-and-spoke pattern
    BRICK = "brick"  # Brick-layer pattern


class AnsatzType(Enum):
    """Rotation layer used in each variational layer.

    STRONGLY_ENTANGLING applies one ``Rot`` per qubit followed by the configured
    entanglement pattern; it is not ``qml.StronglyEntanglingLayers``.
    HARDWARE_EFFICIENT applies ``VQCConfig.rotation_gates`` to every qubit.
    """

    HARDWARE_EFFICIENT = "hardware_efficient"
    STRONGLY_ENTANGLING = "strongly_entangling"


@dataclass
class VQCConfig:
    """Circuit shape and gate choices.

    ``rotation_gates`` is only used by ``AnsatzType.HARDWARE_EFFICIENT``.

    Raises:
        ValueError: If ``n_qubits < 2``, ``n_layers < 1``, or a gate is not one of
            RX, RY, RZ, Rot.
    """

    n_qubits: int = 8
    n_layers: int = 4
    ansatz_type: AnsatzType = AnsatzType.STRONGLY_ENTANGLING
    entanglement: EntanglementPattern = EntanglementPattern.FULL
    rotation_gates: Sequence[str] = field(default_factory=lambda: ["RY", "RZ"])
    data_reuploading: bool = True

    def __post_init__(self) -> None:
        if self.n_qubits < 2:
            raise ValueError("VQC requires at least 2 qubits")
        if self.n_layers < 1:
            raise ValueError("VQC requires at least 1 layer")

        valid_gates = {"RX", "RY", "RZ", "Rot"}
        for gate in self.rotation_gates:
            if gate not in valid_gates:
                raise ValueError(f"Invalid rotation gate: {gate}. Valid: {valid_gates}")


class VariationalQuantumCircuit:
    """Angle-encoded variational circuit returning basis-state probabilities.

    Args:
        config: Circuit configuration.
        device_name: PennyLane device name.
        shots: Shots per evaluation; None for exact probabilities.
        seed: Seed for parameter initialization.
    """

    def __init__(
        self,
        config: VQCConfig,
        device_name: str = "default.qubit",
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self._shots = shots
        self._seed = seed

        # Initialize random number generator
        self._rng = np.random.default_rng(seed)

        # Create PennyLane device
        self.device = qml.device(
            device_name,
            wires=config.n_qubits,
            shots=shots,
        )

        # Calculate parameter shape based on ansatz
        self._param_shape = self._calculate_param_shape()

        # Initialize parameters using He initialization
        self.params = self._initialize_parameters()

        # Build the quantum circuit
        self._circuit = self._build_circuit()

        logger.info(
            f"Initialized VQC with {config.n_qubits} qubits, "
            f"{config.n_layers} layers, {self.num_params} parameters"
        )

    def _calculate_param_shape(self) -> tuple[int, ...]:
        n_qubits = self.config.n_qubits
        n_layers = self.config.n_layers

        if self.config.ansatz_type == AnsatzType.STRONGLY_ENTANGLING:
            # One Rot gate (3 angles) per qubit per layer
            return (n_layers, n_qubits, 3)
        # HARDWARE_EFFICIENT: one angle per single-axis gate, three per Rot
        n_angles = sum(3 if gate == "Rot" else 1 for gate in self.config.rotation_gates)
        return (n_layers, n_qubits, n_angles)

    def _initialize_parameters(self) -> NDArray[np.float64]:
        # He initialization: scale by sqrt(2/n)
        fan_in = np.prod(self._param_shape[1:])
        scale = np.sqrt(2.0 / fan_in)

        params = self._rng.normal(0, scale, self._param_shape)
        return params.astype(np.float64)

    def _build_circuit(self) -> qml.QNode:
        @qml.qnode(self.device, interface="autograd", diff_method="parameter-shift")
        def circuit(
            params: NDArray[np.float64],
            features: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            # Feature encoding using angle embedding
            self._encode_features(features)

            # Apply variational layers
            for layer in range(self.config.n_layers):
                # Data re-uploading (if enabled)
                if self.config.data_reuploading and layer > 0:
                    self._encode_features(features)

                # Apply variational layer
                self._apply_variational_layer(params[layer], layer)

                # Apply entanglement
                self._apply_entanglement(layer)

            # Measurement
            return self._measure()

        return circuit

    def _encode_features(self, features: NDArray[np.float64]) -> None:
        n_features = min(len(features), self.config.n_qubits)

        for i in range(n_features):
            # Angle embedding with RY rotation
            qml.RY(features[i], wires=i)

    def _apply_variational_layer(
        self,
        layer_params: NDArray[np.float64],
        layer_idx: int,
    ) -> None:
        if self.config.ansatz_type == AnsatzType.STRONGLY_ENTANGLING:
            # Apply Rot gates (RZ-RY-RZ decomposition)
            for qubit in range(self.config.n_qubits):
                qml.Rot(
                    layer_params[qubit, 0],
                    layer_params[qubit, 1],
                    layer_params[qubit, 2],
                    wires=qubit,
                )
        else:
            # Apply specified rotation gates
            for qubit in range(self.config.n_qubits):
                offset = 0
                for gate_name in self.config.rotation_gates:
                    gate = getattr(qml, gate_name)
                    if gate_name == "Rot":
                        gate(
                            layer_params[qubit, offset],
                            layer_params[qubit, offset + 1],
                            layer_params[qubit, offset + 2],
                            wires=qubit,
                        )
                        offset += 3
                    else:
                        gate(layer_params[qubit, offset], wires=qubit)
                        offset += 1

    def _apply_entanglement(self, layer_idx: int) -> None:
        n_qubits = self.config.n_qubits
        pattern = self.config.entanglement

        if pattern == EntanglementPattern.LINEAR:
            # Linear chain: (0,1), (1,2), (2,3), ...
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

        elif pattern == EntanglementPattern.CIRCULAR:
            # Ring: Linear + (n-1, 0)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])

        elif pattern == EntanglementPattern.FULL:
            # All-to-all connectivity
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    qml.CNOT(wires=[i, j])

        elif pattern == EntanglementPattern.STAR:
            # Hub-and-spoke (qubit 0 is hub)
            for i in range(1, n_qubits):
                qml.CNOT(wires=[0, i])

        elif pattern == EntanglementPattern.BRICK:
            # Brick layer pattern
            offset = layer_idx % 2
            for i in range(offset, n_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])

    def _measure(self) -> NDArray[np.float64]:
        return qml.probs(wires=range(self.config.n_qubits))

    @property
    def num_params(self) -> int:
        """Total number of trainable angles."""
        return int(np.prod(self._param_shape))

    def forward(
        self,
        features: NDArray[np.float64],
        params: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Evaluate the circuit.

        Args:
            features: 1-D array, zero-padded or truncated to ``n_qubits`` values; each
                value is an RY angle in radians.
            params: Angles with the shape of ``self.params``; defaults to
                ``self.params``.

        Returns:
            Probabilities of shape ``(2**n_qubits,)``.
        """
        if params is None:
            params = self.params

        # Ensure features match qubit count
        if len(features) < self.config.n_qubits:
            features = np.pad(
                features,
                (0, self.config.n_qubits - len(features)),
                mode="constant",
            )
        elif len(features) > self.config.n_qubits:
            features = features[: self.config.n_qubits]

        return self._circuit(params, features)

    def batch_forward(
        self,
        features_batch: NDArray[np.float64],
        params: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Apply `forward` to each row; returns shape ``(batch, 2**n_qubits)``."""
        results = []
        for features in features_batch:
            output = self.forward(features, params)
            results.append(output)
        return np.array(results)

    def get_circuit_depth(self) -> int:
        """Rough depth estimate from the config; ignores data re-uploading."""
        # Approximate depth calculation
        depth_per_layer = 2  # Rotation + entanglement
        if self.config.entanglement == EntanglementPattern.FULL:
            depth_per_layer += self.config.n_qubits - 1

        return self.config.n_layers * depth_per_layer + 1  # +1 for encoding

    def get_gate_count(self) -> dict[str, int]:
        """Gate counts by name, computed from the config rather than the built tape.

        Encoding RY gates are counted once, ignoring re-uploading, and overwrite the
        variational RY count when ``rotation_gates`` contains RY.
        """
        gate_counts: dict[str, int] = {}

        # Rotation gates
        n_rotations = self.config.n_qubits * self.config.n_layers
        if self.config.ansatz_type == AnsatzType.STRONGLY_ENTANGLING:
            gate_counts["Rot"] = n_rotations
        else:
            for gate in self.config.rotation_gates:
                gate_counts[gate] = n_rotations

        # Entanglement gates
        if self.config.entanglement == EntanglementPattern.LINEAR:
            n_cnots = (self.config.n_qubits - 1) * self.config.n_layers
        elif self.config.entanglement == EntanglementPattern.CIRCULAR:
            n_cnots = self.config.n_qubits * self.config.n_layers
        elif self.config.entanglement == EntanglementPattern.FULL:
            n_cnots = (
                self.config.n_qubits * (self.config.n_qubits - 1) // 2
            ) * self.config.n_layers
        elif self.config.entanglement == EntanglementPattern.STAR:
            n_cnots = (self.config.n_qubits - 1) * self.config.n_layers
        else:
            n_cnots = (self.config.n_qubits // 2) * self.config.n_layers

        gate_counts["CNOT"] = n_cnots
        gate_counts["RY"] = self.config.n_qubits  # Feature encoding

        return gate_counts

    def draw(self, style: str = "text") -> Any:
        """Draw the circuit with all-zero features.

        Args:
            style: ``"text"`` returns a string; ``"mpl"`` returns a matplotlib figure
                and needs matplotlib installed.

        Raises:
            ValueError: For any other style.
        """
        # Create a sample execution for drawing
        sample_features = np.zeros(self.config.n_qubits)

        if style == "text":
            return qml.draw(self._circuit)(self.params, sample_features)
        if style == "mpl":
            return qml.draw_mpl(self._circuit)(self.params, sample_features)
        raise ValueError(f"Unknown draw style {style!r}; expected 'text' or 'mpl'")

    def to_dict(self) -> dict[str, Any]:
        """Serialize config and parameters to JSON-compatible types."""
        return {
            "config": {
                "n_qubits": self.config.n_qubits,
                "n_layers": self.config.n_layers,
                "ansatz_type": self.config.ansatz_type.value,
                "entanglement": self.config.entanglement.value,
                "rotation_gates": list(self.config.rotation_gates),
                "data_reuploading": self.config.data_reuploading,
            },
            "params": self.params.tolist(),
            "param_shape": list(self._param_shape),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VariationalQuantumCircuit:
        """Rebuild a circuit from `to_dict` output."""
        config = VQCConfig(
            n_qubits=data["config"]["n_qubits"],
            n_layers=data["config"]["n_layers"],
            ansatz_type=AnsatzType(data["config"]["ansatz_type"]),
            entanglement=EntanglementPattern(data["config"]["entanglement"]),
            rotation_gates=tuple(data["config"]["rotation_gates"]),
            data_reuploading=data["config"]["data_reuploading"],
        )

        circuit = cls(config)
        circuit.params = np.array(data["params"])
        return circuit
