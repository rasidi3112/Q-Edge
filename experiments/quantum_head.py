"""RQ1: a VQC head against a classical head of matched size under non-IID data.

Grid from ``configs/quantum_head.yaml``: datasets, Dirichlet alphas, seeds
and conditions (classical head, noiseless VQC, VQC with depolarizing noise,
and the same with Richardson ZNE). Writes the final test accuracy of every
run, head parameter counts and run times.

With ``--conditions`` only those conditions run, and their rows replace
the matching rows of an existing output file.

Usage:
    python -m experiments.quantum_head [--config PATH] [--conditions NAME ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.common import load_config, mean_std, write_summary
from qedge.fl.simulation import SimulationConfig, run_simulation


def main() -> None:
    """Run the grid and write the summary JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--conditions", nargs="+", help="rerun only these conditions")
    args = parser.parse_args()
    config = load_config("quantum_head", args.config)
    selected = args.conditions or [c["name"] for c in config["conditions"]]
    unknown = set(selected) - {c["name"] for c in config["conditions"]}
    if unknown:
        parser.error(f"unknown conditions {sorted(unknown)}")

    rows: list[dict[str, Any]] = []
    for dataset in config["datasets"]:
        for alpha in config["alphas"]:
            for condition in config["conditions"]:
                if condition["name"] not in selected:
                    continue
                options = dict(config["head"])
                options.update({k: v for k, v in condition.items() if k not in ("name", "model")})
                finals, seconds, head = [], [], None
                for seed in config["seeds"]:
                    summary = run_simulation(
                        SimulationConfig(
                            dataset=dataset,
                            model=condition["model"],
                            clients=config["clients"],
                            rounds=config["rounds"],
                            alpha=alpha,
                            seed=seed,
                            model_options=options,
                        )
                    )
                    finals.append(summary["final_accuracy"])
                    seconds.append(summary["seconds"])
                    head = summary["model"]
                stats = mean_std(finals)
                rows.append(
                    {
                        "dataset": dataset,
                        "alpha": alpha,
                        "condition": condition["name"],
                        "final_accuracy": stats,
                        "per_seed": finals,
                        "seconds_mean": mean_std(seconds)["mean"],
                        "model": head,
                    }
                )
                print(
                    f"{dataset:13} alpha={alpha:<4g} {condition['name']:20} "
                    f"{stats['mean']:.3f} +- {stats['std']:.3f} ({rows[-1]['seconds_mean']:.0f} s)",
                    flush=True,
                )

    output = Path(config["output"])
    if args.conditions and output.exists():
        kept = [
            r for r in json.loads(output.read_text())["results"] if r["condition"] not in selected
        ]
        order = {c["name"]: i for i, c in enumerate(config["conditions"])}
        rows = sorted(
            kept + rows,
            key=lambda r: (
                config["datasets"].index(r["dataset"]),
                config["alphas"].index(r["alpha"]),
                order[r["condition"]],
            ),
        )
    write_summary(
        output,
        {"description": __doc__.splitlines()[0], "results": rows},
        config,
    )


if __name__ == "__main__":
    main()
