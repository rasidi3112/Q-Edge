import json

import numpy as np
import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from qedge.fl.simulation import SimulationConfig, run_simulation
from qedge.fl.strategies import fedavg

SMALL = {"dataset": "breast_cancer", "clients": 5, "rounds": 8, "alpha": 0.5}


def test_history_has_one_row_per_round():
    summary = run_simulation(SimulationConfig(**SMALL))

    assert summary["parameters"] == 62
    assert [row["round"] for row in summary["history"]] == list(range(1, 9))


def test_simulation_is_deterministic_for_a_seed():
    first = run_simulation(SimulationConfig(**SMALL, strategy="fedadam"))
    second = run_simulation(SimulationConfig(**SMALL, strategy="fedadam"))
    other = run_simulation(SimulationConfig(**SMALL, strategy="fedadam", seed=1))

    losses = [[row["test_loss"] for row in s["history"]] for s in (first, second, other)]
    assert losses[0] == losses[1] != losses[2]


def test_simulation_writes_config_and_one_metrics_row_per_round(tmp_path):
    summary = run_simulation(SimulationConfig(**SMALL), tmp_path)

    run_dir = tmp_path / summary["run_dir"].rsplit("/", 1)[-1]
    record = yaml.safe_load((run_dir / "config.yaml").read_text())
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert record["config"]["clients"] == 5
    assert sum(p["samples"] for p in record["partition"]) == 455
    assert record["environment"]["packages"]["torch"]
    assert len(rows) == 8 and rows[-1]["test_accuracy"] == summary["final_accuracy"]


def test_compression_reduces_bytes_up_by_the_kept_fraction():
    dense = run_simulation(SimulationConfig(**SMALL))
    sparse = run_simulation(SimulationConfig(**SMALL, compression=0.1))

    # 62 parameters: 2 tensors keep round(0.1 * 60) + round(0.1 * 2) -> 6 + 1 entries.
    assert sparse["bytes_up"] == 5 * 8 * (6 + 1) * 8
    assert dense["bytes_up"] == 5 * 8 * 62 * 4
    assert sparse["final_accuracy"] > 0.85


def test_custom_aggregator_is_used():
    calls = []

    class Recording:
        def step(self, global_weights, updates):
            calls.append(len(updates))
            return fedavg(updates)

    run_simulation(SimulationConfig(**SMALL), aggregator=Recording())

    assert calls == [5] * 8


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"strategy": "scaffold"}, "Unknown strategy"),
        ({"mu": 0.1}, "fedprox"),
        ({"clients": 0}, "at least 1"),
        ({"dp_noise": 1.0}, "dp_clip"),
    ],
)
def test_invalid_configs_are_rejected(options, message):
    with pytest.raises(ValueError, match=message):
        SimulationConfig(**options)


def test_fedprox_defaults_to_a_positive_mu():
    assert SimulationConfig(strategy="fedprox").train_config().mu == 0.01
    assert SimulationConfig(strategy="fedprox").server_strategy().name == "fedavg"


def test_dp_noise_changes_the_trajectory():
    clean = run_simulation(SimulationConfig(**SMALL, dp_clip=10.0))
    noisy = run_simulation(SimulationConfig(**SMALL, dp_clip=10.0, dp_noise=1.0))

    assert not np.isclose(clean["history"][-1]["test_loss"], noisy["history"][-1]["test_loss"])


def test_fake_samples_break_fedavg_and_median_resists():
    attacked = {"attack": "fake_samples", "attack_fraction": 0.3}

    plain = run_simulation(SimulationConfig(**SMALL, **attacked))
    median = run_simulation(SimulationConfig(**SMALL, **attacked, defense="median"))

    assert plain["final_accuracy"] < 0.5
    assert median["final_accuracy"] > 0.9


def test_attackers_are_recorded(tmp_path):
    config = SimulationConfig(**SMALL, attack="label_flip", attack_fraction=0.4)
    summary = run_simulation(config, tmp_path)

    record = yaml.safe_load(
        (tmp_path / summary["run_dir"].rsplit("/", 1)[-1] / "config.yaml").read_text()
    )
    assert len(record["attackers"]) == 2
    assert record["config"]["attack"] == "label_flip"


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"attack_fraction": 0.2}, "need an attack"),
        ({"defense": "median", "strategy": "fedadam"}, "fedavg or fedprox"),
        ({"defense_options": {"beta": 0.1}}, "need a defense"),
        ({"defense": "median", "defense_options": {"beta": 0.1}}, "do not apply"),
        ({"attack": "backdoor"}, "Unknown attack"),
    ],
)
def test_invalid_attack_and_defense_configs_are_rejected(options, message):
    with pytest.raises(ValueError, match=message):
        SimulationConfig(**options)


def test_secure_aggregation_matches_plain_fedavg():
    plain = run_simulation(SimulationConfig(**{**SMALL, "rounds": 3}))
    secure = run_simulation(SimulationConfig(**{**SMALL, "rounds": 3}, secure_aggregation=True))

    for a, b in zip(plain["history"], secure["history"], strict=True):
        assert b["test_loss"] == pytest.approx(a["test_loss"], abs=1e-4)
        assert b["secagg_bytes_up_per_client"] > a["bytes_up"] / 5


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"defense": "median"}, "cannot run"),
        ({"strategy": "fedadam"}, "computes FedAvg"),
    ],
)
def test_secure_aggregation_excludes_robust_rules_and_adaptive_servers(options, message):
    with pytest.raises(ValueError, match=message):
        SimulationConfig(secure_aggregation=True, **options)
