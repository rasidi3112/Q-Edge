"""Gradient variance, kernel concentration and expressibility of the circuits.

- Gradient variance of the first RY angle for 2-8 qubits, 8 strongly
  entangling layers, global and local Z observables, 200 weight draws.
- Off-diagonal fidelity-kernel mean and variance for 2-8 qubits on the first
  n standardized breast_cancer features (40 samples), angle and zz maps.
- Expressibility (KL to Haar) of both ansatze on 4 qubits for 1-4 layers,
  500 pairs of random weights.

Widths, depths and sample counts come from ``configs/quantum_diagnostics.yaml``.

Usage:
    python -m experiments.quantum_diagnostics [--config PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiments.common import load_config, write_summary
from qedge.fl.partition import load_dataset
from qedge.quantum.circuits import ANSATZE
from qedge.quantum.diagnostics import (
    ansatz_expressibility,
    gradient_variance,
    kernel_concentration,
    scaling,
)
from qedge.quantum.kernels import FEATURE_MAPS, QuantumKernel


def main() -> None:
    """Run all three diagnostics and write the summary JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path)
    config = load_config("quantum_diagnostics", parser.parse_args().config)
    widths = config["widths"]
    grad, kern, expr = config["gradient"], config["kernel"], config["expressibility"]

    gradients: dict[str, Any] = {}
    for observable in ("global", "local"):
        variances = [
            gradient_variance(n, grad["layers"], observable=observable, samples=grad["samples"])
            for n in widths
        ]
        gradients[observable] = {
            "variance": dict(zip(map(str, widths), variances, strict=True)),
            "log_slope_per_2_qubits": scaling(variances),
        }
        print(f"gradient variance ({observable}): {[f'{v:.2e}' for v in variances]}")

    data = load_dataset("breast_cancer", seed=0)
    kernels: dict[str, Any] = {}
    for kind in FEATURE_MAPS:
        kernels[kind] = {
            str(n): kernel_concentration(
                QuantumKernel(n, kind=kind), data.x_train[: kern["samples"], :n]
            )
            for n in widths
        }
        print(f"kernel {kind}: {kernels[kind]}")

    expressibility = {
        ansatz: {
            str(layers): ansatz_expressibility(
                expr["n_qubits"], layers, ansatz, pairs=expr["pairs"]
            )
            for layers in expr["layers"]
        }
        for ansatz in ANSATZE
    }
    print(f"expressibility: {expressibility}")

    write_summary(
        Path(config["output"]),
        {
            "description": __doc__.splitlines()[0],
            "gradient_variance": {**grad, **gradients},
            "kernel_concentration": {**kern, "dataset": "breast_cancer", **kernels},
            "expressibility_kl": {**expr, "bins": 75, **expressibility},
        },
        config,
    )


if __name__ == "__main__":
    main()
