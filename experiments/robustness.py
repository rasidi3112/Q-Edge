"""Final accuracy under malicious clients for each aggregation rule.

Grid from ``configs/robustness.yaml``: (dataset, model) setups, attacks
(`qedge.fl.attacks`), attacker fractions, rules (null is FedAvg; the others
are `qedge.fl.robust.DEFENSES` with default options: trimmed mean beta 0.2,
Krum f = (n - 3) // 2, norm clip at the median norm, sample cap 3x median)
and seeds. With 0% attackers the attack does not matter, so it runs once
per rule.

Usage:
    python -m experiments.robustness [--config PATH]
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
    config = load_config("robustness", parser.parse_args().config)

    rows: list[dict[str, Any]] = []
    for dataset, model in config["setups"]:
        for fraction in config["fractions"]:
            for attack in config["attacks"] if fraction else [None]:
                for rule in config["rules"]:
                    finals = []
                    for seed in config["seeds"]:
                        summary = run_simulation(
                            SimulationConfig(
                                dataset=dataset,
                                model=model,
                                seed=seed,
                                clients=config["clients"],
                                rounds=config["rounds"],
                                alpha=config["alpha"],
                                attack=attack,
                                attack_fraction=fraction,
                                defense=rule,
                            )
                        )
                        finals.append(summary["final_accuracy"])
                    stats = mean_std(finals)
                    rows.append(
                        {
                            "dataset": dataset,
                            "attack": attack,
                            "fraction": fraction,
                            "rule": rule or "fedavg",
                            "final_accuracy": stats,
                            "per_seed": finals,
                        }
                    )
                    print(
                        f"{dataset:13} {attack or '-':12} {fraction:.1f} {rule or 'fedavg':12} "
                        f"{stats['mean']:.3f} +- {stats['std']:.3f}"
                    )

    write_summary(
        Path(config["output"]), {"description": __doc__.splitlines()[0], "results": rows}, config
    )


if __name__ == "__main__":
    main()
