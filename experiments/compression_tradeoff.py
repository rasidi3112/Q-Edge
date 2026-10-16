"""Upload bytes vs final accuracy for top-k compression with error feedback.

Grid from ``configs/compression.yaml``: (dataset, model) setups, keep ratios
(null is dense) and seeds, with FedAvg. A kept entry costs 8 bytes (index
and float32 value), a dense entry 4.

Usage:
    python -m experiments.compression_tradeoff [--config PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiments.common import load_config, mean_std, write_summary
from qedge.fl.simulation import SimulationConfig, run_simulation


def main() -> None:
    """Run every ratio and write the summary JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path)
    config = load_config("compression", parser.parse_args().config)

    rows: list[dict[str, Any]] = []
    for dataset, model in config["setups"]:
        for ratio in config["ratios"]:
            finals, uploads = [], []
            for seed in config["seeds"]:
                summary = run_simulation(
                    SimulationConfig(
                        dataset=dataset,
                        model=model,
                        compression=ratio,
                        seed=seed,
                        clients=config["clients"],
                        rounds=config["rounds"],
                        alpha=config["alpha"],
                    )
                )
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
        Path(config["output"]), {"description": __doc__.splitlines()[0], "results": rows}, config
    )


if __name__ == "__main__":
    main()
