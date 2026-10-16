import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from qedge.cli import ATTACKS as CLI_ATTACKS
from qedge.cli import DEFENSES as CLI_DEFENSES
from qedge.cli import FL_DATASETS, FL_MODELS, FL_STRATEGIES, main
from qedge.fl.attacks import ATTACKS
from qedge.fl.models import MODELS
from qedge.fl.partition import DATASETS
from qedge.fl.robust import DEFENSES
from qedge.fl.simulation import STRATEGIES


def test_cli_choices_match_the_fl_modules():
    assert (FL_DATASETS, FL_MODELS, FL_STRATEGIES) == (DATASETS, MODELS, STRATEGIES)
    assert (CLI_ATTACKS, CLI_DEFENSES) == (ATTACKS, DEFENSES)


def test_cli_simulate_writes_a_run_directory(tmp_path, capsys):
    args = ["simulate", "--clients", "4", "--rounds", "3", "--out", str(tmp_path)]

    assert main(args) == 0

    (run_dir,) = tmp_path.iterdir()
    assert (run_dir / "metrics.jsonl").read_text().count("\n") == 3
    assert "final accuracy:" in capsys.readouterr().out


def test_cli_simulate_reports_invalid_options(tmp_path, capsys):
    assert main(["simulate", "--mu", "0.5", "--out", str(tmp_path)]) == 1
    assert "fedprox" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_cli_simulate_runs_an_attack_against_a_defense(tmp_path, capsys):
    args = ["simulate", "--clients", "5", "--rounds", "3", "--out", str(tmp_path)]
    args += ["--attack", "scaling", "--attack-fraction", "0.2", "--defense", "trimmed_mean"]

    assert main([*args, "--trim", "0.2"]) == 0
    (run_dir,) = tmp_path.iterdir()
    assert run_dir.name.endswith("-scaling0.2-trimmed_mean")
    assert main([*args, "--krum-f", "1"]) == 1
    assert "do not apply" in capsys.readouterr().err
