"""Angle-embedded variational circuits.

Inputs are encoded with one RY rotation per qubit (`qml.AngleEmbedding`).
The trainable block is either `qml.StronglyEntanglingLayers` (Schuld et al.,
"Circuit-centric quantum classifiers", 2018, arXiv:1804.00633) or a
hardware-efficient layer of RY and RZ per qubit followed by a CNOT ring.
With data re-uploading the embedding is repeated before every layer
(Pérez-Salinas et al., "Data re-uploading for a universal quantum
classifier", 2020, arXiv:1907.02085).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pennylane as qml

ANSATZE = ("strongly_entangling", "hardware_efficient")


def weight_shape(ansatz: str, n_layers: int, n_qubits: int) -> tuple[int, int, int]:
    """Shape of the trainable weights for ``ansatz``.

    Raises:
        ValueError: If ``ansatz`` is unknown or a size is below 1.
    """
    if ansatz not in ANSATZE:
        raise ValueError(f"Unknown ansatz {ansatz!r}; expected {ANSATZE}")
    if n_layers < 1 or n_qubits < 1:
        raise ValueError("n_layers and n_qubits must be at least 1")
    return (n_layers, n_qubits, 3 if ansatz == "strongly_entangling" else 2)


def hardware_efficient_layers(weights: Any, wires: Sequence[int]) -> None:
    """RY and RZ on every wire, then CNOTs in a ring (a chain for two wires)."""
    n = len(wires)
    pairs = [(wires[i], wires[(i + 1) % n]) for i in range(n if n > 2 else n - 1)]
    for layer in range(weights.shape[0]):
        for i, wire in enumerate(wires):
            qml.RY(weights[layer, i, 0], wires=wire)
            qml.RZ(weights[layer, i, 1], wires=wire)
        for control, target in pairs:
            qml.CNOT(wires=[control, target])


def variational_block(weights: Any, wires: Sequence[int], ansatz: str) -> None:
    """Apply the trainable layers of ``ansatz``."""
    if ansatz == "strongly_entangling":
        qml.StronglyEntanglingLayers(weights, wires=wires)
    else:
        hardware_efficient_layers(weights, wires)


def vqc(inputs: Any, weights: Any, *, ansatz: str, reupload: bool) -> list[Any]:
    """Embed ``inputs`` (one angle per qubit), apply the layers, measure Z.

    Args:
        inputs: Angles of shape ``(n_qubits,)`` or ``(batch, n_qubits)``.
        weights: Array of shape `weight_shape`.
        ansatz: One of `ANSATZE`.
        reupload: Repeat the embedding before every layer.

    Returns:
        ``<Z_i>`` for each qubit.
    """
    wires = list(range(weights.shape[1]))
    if reupload:
        for layer in range(weights.shape[0]):
            qml.AngleEmbedding(inputs, wires=wires, rotation="Y")
            variational_block(weights[layer : layer + 1], wires, ansatz)
    else:
        qml.AngleEmbedding(inputs, wires=wires, rotation="Y")
        variational_block(weights, wires, ansatz)
    return [qml.expval(qml.PauliZ(w)) for w in wires]


def probabilities_circuit(inputs: Any, weights: Any, *, ansatz: str) -> Any:
    """Embed ``inputs``, apply the layers and return basis-state probabilities."""
    wires = list(range(weights.shape[1]))
    qml.AngleEmbedding(inputs, wires=wires, rotation="Y")
    variational_block(weights, wires, ansatz)
    return qml.probs(wires=wires)
