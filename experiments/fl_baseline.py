"""Final test accuracy of each strategy under Dirichlet label skew.

Grid: datasets breast_cancer (logreg) and digits (mlp), strategies fedavg,
fedprox, fedadam and fedyogi, alpha in {0.1, 0.5, 100}, 10 clients, 30
rounds, one local epoch, seeds 0-2. Writes results/summary/fl_baseline.json.

Usage:
    python -m experiments.fl_baseline [--seeds 0 1 2] [--output PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiments.common import mean_std, write_summary
from qedge.fl.simulation import STRATEGIES, SimulationConfig, run_simulation

SETUPS = (("breast_cancer", "logreg"), ("digits", "mlp"))
ALPHAS = (0.1, 0.5, 100.0)


def main() -> None:
    """Run the grid and write the summary JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, default=Path("results/summary/fl_baseline.json"))
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for dataset, model in SETUPS:
        for strategy in STRATEGIES:
            for alpha in ALPHAS:
                finals, seconds = [], []
                for seed in args.seeds:
                    config = SimulationConfig(
                        dataset=dataset, model=model, strategy=strategy, alpha=alpha, seed=seed
                    )
                    summary = run_simulation(config)
                    finals.append(summary["final_accuracy"])
                    seconds.append(summary["seconds"])
                stats = mean_std(finals)
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "strategy": strategy,
                        "alpha": alpha,
                        "final_accuracy": stats,
                        "per_seed": finals,
                        "seconds_mean": mean_std(seconds)["mean"],
                    }
                )
                print(
                    f"{dataset:13} {strategy:8} alpha={alpha:<5g} "
                    f"{stats['mean']:.4f} +- {stats['std']:.4f}"
                )

    write_summary(
        args.output,
        {
            "description": __doc__.splitlines()[0],
            "setup": {
                "clients": 10,
                "rounds": 30,
                "local_epochs": 1,
                "client_lr": 0.1,
                "server_lr": 0.05,
                "fedprox_mu": 0.01,
                "seeds": args.seeds,
            },
            "results": rows,
        },
    )


if __name__ == "__main__":
    main()
