"""Hybrid quantum-classical classifier and its classical counterpart.

Both models share the same encoder and readout and differ only in the head:

- `HybridVQCClassifier`: features -> MLP encoder -> ``pi * tanh`` angles ->
  VQC (`qedge.quantum.circuits.vqc`) -> ``<Z_i>`` per qubit -> linear readout.
- `ClassicalHeadClassifier`: the VQC is replaced by ``Linear(n, n) -> tanh``
  or ``Linear(n, h) -> tanh -> Linear(h, n) -> tanh`` with ``h >= n``,
  whichever has a parameter count closest to the VQC's.

The VQC runs through `qml.qnn.TorchLayer`. Without shots it is
differentiated by backpropagation through the simulator, also on the
noisy ``default.mixed`` device, which gives exact gradients of the noisy
model; with shots it uses the parameter-shift rule, as hardware would.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised in a subprocess test
    raise ImportError("PyTorch is required for qedge.quantum.models; install q-edge[fl]") from exc

import pennylane as qml

from qedge.quantum.circuits import ANSATZE, vqc, weight_shape
from qedge.quantum.noise import NoiseConfig, make_device
from qedge.quantum.zne import ZNEConfig, achieved_scale_factors, mitigate


@dataclass(frozen=True)
class QuantumHeadConfig:
    """Shape and noise of the quantum head.

    Attributes:
        n_qubits: Width of the VQC; the encoder outputs one angle per qubit.
        n_layers: Variational layers.
        ansatz: One of `qedge.quantum.circuits.ANSATZE`.
        reupload: Repeat the embedding before every layer.
        noise: Gate noise, readout noise and shots.
        zne: Zero-noise extrapolation of every forward pass; None disables it.
        encoder_hidden: Hidden width of the MLP encoder.

    Raises:
        ValueError: For an unknown ansatz, fewer than 1 qubit or layer, or ZNE
            without gate noise.
    """

    n_qubits: int = 4
    n_layers: int = 2
    ansatz: str = "strongly_entangling"
    reupload: bool = False
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    zne: ZNEConfig | None = None
    encoder_hidden: int = 8

    def __post_init__(self) -> None:
        if self.ansatz not in ANSATZE:
            raise ValueError(f"Unknown ansatz {self.ansatz!r}; expected {ANSATZE}")
        if self.n_qubits < 1 or self.n_layers < 1:
            raise ValueError("n_qubits and n_layers must be at least 1")
        if self.zne is not None and self.noise.p == 0:
            raise ValueError("ZNE needs gate noise (noise.p > 0) to mitigate")

    @property
    def vqc_parameters(self) -> int:
        """Trainable angles in the VQC."""
        layers, qubits, angles = weight_shape(self.ansatz, self.n_layers, self.n_qubits)
        return layers * qubits * angles


def head_config(
    *,
    n_qubits: int = 4,
    n_layers: int = 2,
    ansatz: str = "strongly_entangling",
    reupload: bool = False,
    noise: float = 0.0,
    noise_channel: str = "depolarizing",
    readout: float = 0.0,
    shots: int | None = None,
    zne: str | None = None,
    zne_scales: list[float] | None = None,
    zne_order: int = 1,
    encoder_hidden: int = 8,
) -> QuantumHeadConfig:
    """Build a `QuantumHeadConfig` from flat options, as stored in run configs."""
    zne_config = None
    if zne is not None:
        scales = tuple(float(s) for s in (zne_scales or (1.0, 3.0, 5.0)))
        zne_config = ZNEConfig(method=zne, scale_factors=scales, order=zne_order)
    return QuantumHeadConfig(
        n_qubits=n_qubits,
        n_layers=n_layers,
        ansatz=ansatz,
        reupload=reupload,
        noise=NoiseConfig(p=noise, channel=noise_channel, readout=readout, shots=shots),
        zne=zne_config,
        encoder_hidden=encoder_hidden,
    )


def achieved_zne_factors(config: QuantumHeadConfig) -> list[float] | None:
    """Noise factors that folding reaches on this head's circuit; None without ZNE."""
    if config.zne is None:
        return None
    shape = weight_shape(config.ansatz, config.n_layers, config.n_qubits)
    tape = qml.tape.make_qscript(vqc)(
        torch.zeros(config.n_qubits),
        torch.zeros(shape),
        ansatz=config.ansatz,
        reupload=config.reupload,
    )
    tape = qml.transforms.decompose(tape, gate_set={"RY", "RZ", "Rot", "CNOT"})[0][0]
    return achieved_scale_factors(tape, config.zne.scale_factors)


def build_vqc_layer(config: QuantumHeadConfig, *, seed: int | None = None) -> nn.Module:
    """The VQC as a `qml.qnn.TorchLayer` mapping ``(batch, n_qubits)`` angles to ``<Z>``."""
    device = make_device(config.n_qubits, config.noise, seed=seed)
    diff_method = "backprop" if config.noise.shots is None else "parameter-shift"

    def circuit(inputs: Any, weights: Any) -> Any:
        return vqc(inputs, weights, ansatz=config.ansatz, reupload=config.reupload)

    qnode: Any = qml.QNode(circuit, device, interface="torch", diff_method=diff_method)
    if config.noise.shots is not None:
        qnode = qml.set_shots(qnode, shots=config.noise.shots)
    if config.zne is not None:
        qnode = mitigate(qnode, config.zne)
    shape = weight_shape(config.ansatz, config.n_layers, config.n_qubits)
    layer: nn.Module = qml.qnn.TorchLayer(qnode, {"weights": shape})
    return layer


def matched_head_shape(n_qubits: int, target: int) -> int | None:
    """Hidden width of the classical head closest to ``target`` parameters.

    Candidates are one ``Linear(n, n)`` layer (``n^2 + n`` parameters,
    returned as None) and two layers through a hidden width ``h >= n``
    (``h (2n + 1) + n``). Widths below ``n`` are excluded so the classical
    head is never a narrower bottleneck than the n expectation values of the
    VQC. Ties go to the single layer.
    """
    n = n_qubits
    best: tuple[int, int | None] = (abs(n * n + n - target), None)
    hidden = max(n, round((target - n) / (2 * n + 1)))
    for h in {n, hidden, hidden + 1}:
        if h >= n:
            best = min(best, (abs(h * (2 * n + 1) + n - target), h), key=lambda c: c[0])
    return best[1]


class _Classifier(nn.Module):
    def __init__(
        self, n_features: int, n_classes: int, config: QuantumHeadConfig, head: nn.Module
    ) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(n_features, config.encoder_hidden),
            nn.ReLU(),
            nn.Linear(config.encoder_hidden, config.n_qubits),
            nn.Tanh(),
        )
        self.head = head
        self.readout = nn.Linear(config.n_qubits, n_classes)

    def describe(self) -> dict[str, Any]:
        """Parameter counts and achieved ZNE factors for run records."""
        return {
            "head_parameters": sum(int(p.numel()) for p in self.head.parameters()),
            "vqc_parameters": self.config.vqc_parameters,
            "zne_achieved_scale_factors": achieved_zne_factors(self.config),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = torch.pi * self.encoder(x)
        logits: torch.Tensor = self.readout(self.head(angles).to(angles.dtype))
        return logits


class HybridVQCClassifier(_Classifier):
    """Encoder, VQC head and linear readout; see the module docstring."""

    def __init__(
        self, n_features: int, n_classes: int, config: QuantumHeadConfig, *, seed: int | None = None
    ) -> None:
        super().__init__(n_features, n_classes, config, build_vqc_layer(config, seed=seed))


class ClassicalHeadClassifier(_Classifier):
    """Same encoder and readout with a tanh head of matched size (`matched_head_shape`)."""

    def __init__(self, n_features: int, n_classes: int, config: QuantumHeadConfig) -> None:
        n = config.n_qubits
        hidden = matched_head_shape(n, config.vqc_parameters)
        head: nn.Module
        if hidden is None:
            head = nn.Sequential(nn.Linear(n, n), nn.Tanh())
        else:
            head = nn.Sequential(nn.Linear(n, hidden), nn.Tanh(), nn.Linear(hidden, n), nn.Tanh())
        super().__init__(n_features, n_classes, config, head)


def build_head_model(
    name: str, n_features: int, n_classes: int, config: QuantumHeadConfig, *, seed: int
) -> nn.Module:
    """Build ``"hybrid-vqc"`` or ``"classical-head"`` with seeded initialization.

    Raises:
        ValueError: For another ``name``.
    """
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        if name == "hybrid-vqc":
            return HybridVQCClassifier(n_features, n_classes, config, seed=seed)
        if name == "classical-head":
            return ClassicalHeadClassifier(n_features, n_classes, config)
        raise ValueError(f"Unknown head model {name!r}")
    finally:
        torch.random.set_rng_state(state)
