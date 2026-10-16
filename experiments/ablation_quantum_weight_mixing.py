"""Reproduce the accuracy cost of the removed "quantum" weight mixing (finding F1).

Before 0.2.0, ``QuantumGlobalAggregator`` replaced 30% of every aggregated
weight with values unrelated to the data: it squeezed the flattened FedAvg
result into 8 numbers, ran them through an 8-qubit variational circuit with
random parameters, took ``log(p)`` of the first 8 basis-state probabilities,
scaled them to ``[-1, 0]`` and spread them back over the weight vector in
blocks. This script re-creates that rule and compares it with plain FedAvg on
the same partitions and seeds.

Writes results/summary/ablation_quantum_weight_mixing.json (or ``--output``).

Usage:
    python -m experiments.ablation_quantum_weight_mixing [--seeds 0 1 2] [--output PATH]
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pennylane as qml
from numpy.typing import NDArray

from experiments.common import mean_std, write_summary
from qedge.fl.simulation import SimulationConfig, run_simulation
from qedge.fl.strategies import Weights, fedavg

CLASSICAL_WEIGHT = 0.7
N_QUBITS = 8


N_LAYERS = 4


def old_circuit(seed: int) -> Any:
    """Rebuild the removed ``VariationalQuantumCircuit`` with its defaults.

    8 qubits, 4 layers, RY re-uploading, one Rot per qubit, all-to-all CNOTs,
    weights from normal(0, sqrt(2/24)); returns basis-state probabilities.
    """
    weights = np.random.default_rng(seed).normal(
        0, np.sqrt(2.0 / (N_QUBITS * 3)), (N_LAYERS, N_QUBITS, 3)
    )

    @qml.qnode(qml.device("default.qubit", wires=N_QUBITS))
    def circuit(features: Any) -> Any:
        for layer in range(N_LAYERS):
            for i in range(N_QUBITS):
                qml.RY(features[i], wires=i)
            for q in range(N_QUBITS):
                qml.Rot(*weights[layer, q], wires=q)
            for i in range(N_QUBITS):
                for j in range(i + 1, N_QUBITS):
                    qml.CNOT(wires=[i, j])
        return qml.probs(wires=range(N_QUBITS))

    return circuit


class QuantumWeightMixing:
    """The pre-0.2.0 hybrid aggregation rule, kept only for this ablation."""

    def __init__(self, seed: int) -> None:
        self.circuit = old_circuit(seed)

    def _compress(self, flat: NDArray[np.float64]) -> NDArray[np.float64]:
        stride = max(len(flat) // N_QUBITS, 1)
        compressed = np.zeros(N_QUBITS)
        for i in range(min(N_QUBITS, len(flat))):
            compressed[i] = np.mean(flat[i * stride : (i + 1) * stride])
        if np.std(compressed) > 0:
            compressed = (compressed - np.mean(compressed)) / np.std(compressed)
        return np.tanh(compressed) * np.pi

    def _expand(self, encoding: NDArray[np.float64], size: int) -> NDArray[np.float64]:
        expanded = np.zeros(size)
        stride = size // len(encoding)
        for i, value in enumerate(encoding):
            expanded[i * stride : (i + 1) * stride] = value
        return expanded

    def step(
        self, global_weights: Weights, updates: Sequence[tuple[Weights, int]]
    ) -> dict[str, NDArray[Any]]:
        """FedAvg, then overwrite 30% of the result with the circuit output."""
        averaged = fedavg(updates)
        names = list(averaged)
        flat = np.concatenate([averaged[n].ravel() for n in names]).astype(np.float64)
        log_probs = np.log(np.asarray(self.circuit(self._compress(flat))) + 1e-10)
        encoding = (log_probs / np.max(np.abs(log_probs)))[:N_QUBITS]
        mixed = CLASSICAL_WEIGHT * flat + (1 - CLASSICAL_WEIGHT) * self._expand(encoding, len(flat))
        result, offset = {}, 0
        for name in names:
            size = averaged[name].size
            chunk = mixed[offset : offset + size].reshape(averaged[name].shape)
            result[name] = chunk.astype(averaged[name].dtype)
            offset += size
        return result


def main() -> None:
    """Run both rules for each seed and write the summary JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--output", type=Path, default=Path("results/summary/ablation_quantum_weight_mixing.json")
    )
    args = parser.parse_args()

    runs: dict[str, list[dict[str, Any]]] = {"fedavg": [], "quantum_weight_mixing": []}
    for seed in args.seeds:
        config = SimulationConfig(
            dataset="breast_cancer", model="logreg", clients=5, rounds=15, alpha=0.5, seed=seed
        )
        for rule, aggregator in (
            ("fedavg", None),
            ("quantum_weight_mixing", QuantumWeightMixing(seed)),
        ):
            summary = run_simulation(config, aggregator=aggregator)
            runs[rule].append(
                {
                    "seed": seed,
                    "final_accuracy": summary["final_accuracy"],
                    "accuracy_per_round": [row["test_accuracy"] for row in summary["history"]],
                }
            )
            print(f"seed {seed} {rule:22} final accuracy {summary['final_accuracy']:.4f}")

    result = {
        "description": __doc__.splitlines()[0],
        "setup": {
            "dataset": "breast_cancer",
            "model": "logreg",
            "clients": 5,
            "rounds": 15,
            "alpha": 0.5,
            "seeds": args.seeds,
            "classical_weight": CLASSICAL_WEIGHT,
            "n_qubits": N_QUBITS,
        },
        "final_accuracy": {
            rule: mean_std([r["final_accuracy"] for r in rows]) for rule, rows in runs.items()
        },
        "runs": runs,
    }
    write_summary(args.output, result)


if __name__ == "__main__":
    main()
