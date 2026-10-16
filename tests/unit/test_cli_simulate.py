import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from qedge.cli import FL_DATASETS, FL_MODELS, FL_STRATEGIES, main
from qedge.fl.models import MODELS
from qedge.fl.partition import DATASETS
from qedge.fl.simulation import STRATEGIES


def test_cli_choices_match_the_fl_modules():
    assert (FL_DATASETS, FL_MODELS, FL_STRATEGIES) == (DATASETS, MODELS, STRATEGIES)


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
