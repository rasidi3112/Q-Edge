"""Error of noisy and ZNE-mitigated expectation values on random circuits.

20 random 4-qubit circuits (angle embedding of random inputs, 2 strongly
entangling layers), observable Z0 Z1, depolarizing and amplitude damping
noise with p in {0.001, 0.01}, scale factors 1, 3, 5 with Richardson,
quadratic and exponential extrapolation. The error is the absolute
difference from the noiseless value. The number of circuits and the noise
probabilities come from ``configs/zne.yaml``.

Usage:
    python -m experiments.zne_accuracy [--config PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pennylane as qml

from experiments.common import load_config, mean_std, write_summary
from qedge.quantum.circuits import variational_block, weight_shape
from qedge.quantum.noise import CHANNELS, NoiseConfig, make_device
from qedge.quantum.zne import ZNEConfig, estimate

N_QUBITS = 4
CONFIGS = {
    "richardson": ZNEConfig("richardson", (1.0, 3.0, 5.0)),
    "poly2": ZNEConfig("poly", (1.0, 3.0, 5.0), order=2),
    "exponential": ZNEConfig("exponential", (1.0, 3.0, 5.0)),
}


def circuit(inputs: Any, weights: Any) -> Any:
    """Angle embedding, strongly entangling layers, then ``<Z0 Z1>``."""
    wires = list(range(N_QUBITS))
    qml.AngleEmbedding(inputs, wires=wires, rotation="Y")
    variational_block(weights, wires, "strongly_entangling")
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


def main() -> None:
    """Measure unmitigated and mitigated errors and write the summary JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path)
    config = load_config("zne", parser.parse_args().config)

    rng = np.random.default_rng(0)
    shape = weight_shape("strongly_entangling", 2, N_QUBITS)
    draws = [
        (rng.uniform(-np.pi, np.pi, N_QUBITS), rng.uniform(0, 2 * np.pi, shape))
        for _ in range(config["circuits"])
    ]
    ideal_qnode = qml.QNode(circuit, make_device(N_QUBITS, NoiseConfig()))
    ideal = [float(ideal_qnode(x, w)) for x, w in draws]

    rows = []
    for channel in CHANNELS:
        for p in config["probabilities"]:
            noisy = qml.QNode(circuit, make_device(N_QUBITS, NoiseConfig(p=p, channel=channel)))
            errors: dict[str, list[float]] = {"unmitigated": [], **{m: [] for m in CONFIGS}}
            for (x, w), target in zip(draws, ideal, strict=True):
                for method, zne_config in CONFIGS.items():
                    result = estimate(noisy, (x, w), zne_config)
                    errors[method].append(abs(result.mitigated - target))
                errors["unmitigated"].append(abs(result.unmitigated - target))
            row = {
                "channel": channel,
                "p": p,
                "abs_error": {name: mean_std(values) for name, values in errors.items()},
            }
            rows.append(row)
            summary = {k: round(v["mean"], 5) for k, v in row["abs_error"].items()}
            print(f"{channel:18} p={p:<6g} {summary}")

    write_summary(
        Path(config["output"]),
        {
            "description": __doc__.splitlines()[0],
            "setup": {
                "n_qubits": N_QUBITS,
                "layers": 2,
                "circuits": config["circuits"],
                "observable": "Z0 Z1",
                "scale_factors": [1, 3, 5],
                "folding": "global",
            },
            "results": rows,
        },
        config,
    )


if __name__ == "__main__":
    main()
