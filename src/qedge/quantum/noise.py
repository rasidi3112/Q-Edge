"""Gate and readout noise for simulated circuits.

Gate noise is a `qml.NoiseModel` attached to the device, so it acts after
QNode transforms such as unitary folding: a folded circuit is noisier in
proportion to its gate count, which zero-noise extrapolation relies on.
Readout noise is a device transform that appends one bit flip per wire after
the last gate, so folding does not amplify it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pennylane as qml
from pennylane.tape import QuantumScript

CHANNELS = ("depolarizing", "amplitude_damping")


@dataclass(frozen=True)
class NoiseConfig:
    """Noise applied to every gate and, optionally, before measurement.

    Attributes:
        p: Gate error probability: depolarizing probability or amplitude
            damping gamma, applied to each wire of every gate.
        channel: One of `CHANNELS`.
        readout: Bit-flip probability on each wire just before measurement.
        shots: Shots per evaluation; None for exact expectation values.

    Raises:
        ValueError: If a probability is outside [0, 1), ``channel`` is unknown
            or ``shots`` is not positive.
    """

    p: float = 0.0
    channel: str = "depolarizing"
    readout: float = 0.0
    shots: int | None = None

    def __post_init__(self) -> None:
        if not (0 <= self.p < 1 and 0 <= self.readout < 1):
            raise ValueError("noise probabilities must be in [0, 1)")
        if self.channel not in CHANNELS:
            raise ValueError(f"Unknown channel {self.channel!r}; expected {CHANNELS}")
        if self.shots is not None and self.shots < 1:
            raise ValueError("shots must be positive")

    @property
    def noiseless(self) -> bool:
        """True when no channel is inserted."""
        return self.p == 0 and self.readout == 0


def gate_noise_model(config: NoiseConfig, n_qubits: int) -> qml.NoiseModel:
    """Noise model that follows every gate with the channel on each of its wires."""
    channel = qml.DepolarizingChannel if config.channel == "depolarizing" else qml.AmplitudeDamping

    def gate_noise(op: Any, **_: Any) -> None:
        for wire in op.wires:
            channel(config.p, wires=wire)

    return qml.NoiseModel({qml.noise.wires_in(range(n_qubits)): gate_noise})


@qml.transform
def readout_flips(tape: QuantumScript, p: float, wires: range) -> tuple[list[QuantumScript], Any]:
    """Append ``BitFlip(p)`` on every wire in ``wires`` after the last gate."""
    ops = [*tape.operations, *(qml.BitFlip(p, wires=w) for w in wires)]
    return [QuantumScript(ops, tape.measurements, shots=tape.shots)], lambda results: results[0]


def make_device(n_qubits: int, config: NoiseConfig, *, seed: int | None = None) -> Any:
    """``default.qubit`` without noise, else ``default.mixed`` with the noise attached."""
    if config.noiseless:
        return qml.device("default.qubit", wires=n_qubits, seed=seed)
    device = qml.device("default.mixed", wires=n_qubits, seed=seed)
    if config.p:
        device = qml.add_noise(device, gate_noise_model(config, n_qubits))
    if config.readout:
        device = readout_flips(device, p=config.readout, wires=range(n_qubits))
    return device
