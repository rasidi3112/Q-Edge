"""Final accuracy of each strategy under Dirichlet label skew.

Grid from ``configs/fl_baseline.yaml``: (dataset, model) setups, strategies,
alphas and seeds; one local epoch, client lr 0.1, server lr 0.05, FedProx mu
0.01.

Usage:
    python -m experiments.fl_baseline [--config PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiments.common import load_config, mean_std, write_summary
from qedge.fl.simulation import SimulationConfig, run_simulation


def main() -> None:
    """Run the grid and write the summary JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path)
    config = load_config("fl_baseline", parser.parse_args().config)

    rows: list[dict[str, Any]] = []
    for dataset, model in config["setups"]:
        for strategy in config["strategies"]:
            for alpha in config["alphas"]:
                finals, seconds = [], []
                for seed in config["seeds"]:
                    summary = run_simulation(
                        SimulationConfig(
                            dataset=dataset,
                            model=model,
                            strategy=strategy,
                            alpha=alpha,
                            seed=seed,
                            clients=config["clients"],
                            rounds=config["rounds"],
                        )
                    )
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
        Path(config["output"]), {"description": __doc__.splitlines()[0], "results": rows}, config
    )


if __name__ == "__main__":
    main()
