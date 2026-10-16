"""Final accuracy under malicious clients for each aggregation rule.

Grid: breast_cancer (logreg) and digits (MLP), attacks label_flip, scaling,
random_noise and fake_samples (`qedge.fl.attacks`) by 0, 10, 20 or 30% of 10
clients, rules FedAvg and every `qedge.fl.robust.DEFENSES` with default
options (trimmed mean beta 0.2, Krum f = 3, norm clip at the median norm,
sample cap 3x median), 30 rounds, alpha 0.5, seeds 0-2. With 0% attackers
the attack does not matter, so it is run once per rule. Writes
results/summary/robustness.json.

Usage:
    python -m experiments.robustness [--seeds 0 1 2] [--output PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiments.common import mean_std, write_summary
from qedge.fl.attacks import ATTACKS
from qedge.fl.robust import DEFENSES
from qedge.fl.simulation import SimulationConfig, run_simulation

SETUPS = (("breast_cancer", "logreg"), ("digits", "mlp"))
FRACTIONS = (0.0, 0.1, 0.2, 0.3)
RULES: tuple[str | None, ...] = (None, *DEFENSES)


def main() -> None:
    """Run the grid and write the summary JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, default=Path("results/summary/robustness.json"))
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for dataset, model in SETUPS:
        for fraction in FRACTIONS:
            for attack in ATTACKS if fraction else (None,):
                for rule in RULES:
                    finals = []
                    for seed in args.seeds:
                        config = SimulationConfig(
                            dataset=dataset,
                            model=model,
                            seed=seed,
                            attack=attack,
                            attack_fraction=fraction,
                            defense=rule,
                        )
                        finals.append(run_simulation(config)["final_accuracy"])
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
        args.output,
        {
            "description": __doc__.splitlines()[0],
            "setup": {
                "clients": 10,
                "rounds": 30,
                "alpha": 0.5,
                "seeds": args.seeds,
                "attack_scales": {
                    "scaling": "number of clients",
                    "random_noise": 1.0,
                    "fake_samples": 100,
                },
                "defense_options": {
                    "trimmed_mean": {"beta": 0.2},
                    "krum": {"f": 3},
                    "multi_krum": {"f": 3, "m": 7},
                    "norm_clip": {"clip": "median delta norm"},
                    "bounded": {"cap": 3.0},
                },
            },
            "results": rows,
        },
    )


if __name__ == "__main__":
    main()
