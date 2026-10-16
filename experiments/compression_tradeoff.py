"""Upload bytes vs final accuracy for top-k compression with error feedback.

FedAvg, 10 clients, 30 rounds, alpha 0.5, seeds 0-2, on breast_cancer
(logreg) and digits (mlp), with keep ratios 1 (dense), 0.5, 0.2, 0.1, 0.05
and 0.01. Writes results/summary/compression.json.

Usage:
    python -m experiments.compression_tradeoff [--seeds 0 1 2] [--output PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiments.common import mean_std, write_summary
from qedge.fl.simulation import SimulationConfig, run_simulation

SETUPS = (("breast_cancer", "logreg"), ("digits", "mlp"))
RATIOS: tuple[float | None, ...] = (None, 0.5, 0.2, 0.1, 0.05, 0.01)


def main() -> None:
    """Run every ratio and write the summary JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, default=Path("results/summary/compression.json"))
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for dataset, model in SETUPS:
        for ratio in RATIOS:
            finals, uploads = [], []
            for seed in args.seeds:
                config = SimulationConfig(
                    dataset=dataset, model=model, compression=ratio, seed=seed
                )
                summary = run_simulation(config)
                finals.append(summary["final_accuracy"])
                uploads.append(summary["bytes_up"])
            stats = mean_std(finals)
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "ratio": 1.0 if ratio is None else ratio,
                    "bytes_up": uploads[0],
                    "final_accuracy": stats,
                    "per_seed": finals,
                }
            )
            print(
                f"{dataset:13} ratio={rows[-1]['ratio']:<5g} bytes_up={uploads[0]:>9} "
                f"{stats['mean']:.4f} +- {stats['std']:.4f}"
            )

    write_summary(
        args.output,
        {
            "description": __doc__.splitlines()[0],
            "setup": {
                "strategy": "fedavg",
                "clients": 10,
                "rounds": 30,
                "alpha": 0.5,
                "seeds": args.seeds,
                "bytes_per_kept_entry": 8,
                "bytes_per_dense_entry": 4,
            },
            "results": rows,
        },
    )


if __name__ == "__main__":
    main()
