"""Gradient variance, kernel concentration and expressibility of the circuits.

- Gradient variance of the first RY angle for 2-8 qubits, 8 strongly
  entangling layers, global and local Z observables, 200 weight draws.
- Off-diagonal fidelity-kernel mean and variance for 2-8 qubits on the first
  n standardized breast_cancer features (40 samples), angle and zz maps.
- Expressibility (KL to Haar) of both ansatze on 4 qubits for 1-4 layers,
  500 pairs of random weights.

Writes results/summary/quantum_diagnostics.json.

Usage:
    python -m experiments.quantum_diagnostics [--output PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiments.common import write_summary
from qedge.fl.partition import load_dataset
from qedge.quantum.circuits import ANSATZE
from qedge.quantum.diagnostics import (
    ansatz_expressibility,
    gradient_variance,
    kernel_concentration,
    scaling,
)
from qedge.quantum.kernels import FEATURE_MAPS, QuantumKernel

WIDTHS = (2, 4, 6, 8)


def main() -> None:
    """Run all three diagnostics and write the summary JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output", type=Path, default=Path("results/summary/quantum_diagnostics.json")
    )
    args = parser.parse_args()

    gradients: dict[str, Any] = {}
    for observable in ("global", "local"):
        variances = [gradient_variance(n, 8, observable=observable, samples=200) for n in WIDTHS]
        gradients[observable] = {
            "variance": dict(zip(map(str, WIDTHS), variances, strict=True)),
            "log_slope_per_2_qubits": scaling(variances),
        }
        print(f"gradient variance ({observable}): {[f'{v:.2e}' for v in variances]}")

    data = load_dataset("breast_cancer", seed=0)
    kernels: dict[str, Any] = {}
    for kind in FEATURE_MAPS:
        kernels[kind] = {
            str(n): kernel_concentration(QuantumKernel(n, kind=kind), data.x_train[:40, :n])
            for n in WIDTHS
        }
        print(f"kernel {kind}: {kernels[kind]}")

    expressibility = {
        ansatz: {str(layers): ansatz_expressibility(4, layers, ansatz) for layers in (1, 2, 3, 4)}
        for ansatz in ANSATZE
    }
    print(f"expressibility: {expressibility}")

    write_summary(
        args.output,
        {
            "description": __doc__.splitlines()[0],
            "gradient_variance": {"layers": 8, "samples": 200, **gradients},
            "kernel_concentration": {"samples": 40, "dataset": "breast_cancer", **kernels},
            "expressibility_kl": {"n_qubits": 4, "pairs": 500, "bins": 75, **expressibility},
        },
    )


if __name__ == "__main__":
    main()
