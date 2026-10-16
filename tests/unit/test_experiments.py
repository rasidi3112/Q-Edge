import json
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from experiments.common import CONFIG_DIR, load_config, mean_std
from experiments.run_all import STEPS, command


def test_every_config_loads_and_names_its_output():
    for path in CONFIG_DIR.glob("*.yaml"):
        config = load_config(path.stem)
        if path.stem == "benchmarks":
            assert {"crypto", "secagg"} <= set(config)
        else:
            assert config["output"].startswith("results/summary/")


def test_run_all_builds_a_command_for_every_step():
    for step in STEPS:
        assert command(step)[0] == sys.executable
    assert command("bench_secagg")[-4:] == ["5", "10", "20", "50"]
    assert command("robustness")[-1] == "experiments.robustness"


def test_mean_std_of_one_value_has_zero_std():
    assert mean_std([0.5]) == {"mean": 0.5, "std": 0.0}


def test_report_writes_sections_and_marks_missing_summaries(tmp_path):
    pytest.importorskip("matplotlib")
    from experiments import report

    summary = tmp_path / "summary"
    summary.mkdir()
    rq1 = {
        "config": {
            "datasets": ["breast_cancer"],
            "alphas": [0.1],
            "seeds": [0],
            "clients": 2,
            "rounds": 1,
            "head": {"n_qubits": 2, "n_layers": 1},
            "conditions": [{"name": "classical-head"}, {"name": "vqc"}],
        },
        "results": [
            {
                "dataset": "breast_cancer",
                "alpha": 0.1,
                "condition": name,
                "final_accuracy": {"mean": 0.9, "std": 0.01},
                "seconds_mean": 1.0,
                "model": {"head_parameters": 6},
            }
            for name in ("classical-head", "vqc")
        ],
    }
    (summary / "quantum_head.json").write_text(json.dumps(rq1))
    docs = tmp_path / "docs"

    argv = sys.argv
    sys.argv = ["report", "--summary", str(summary), "--docs", str(docs)]
    try:
        report.main()
    finally:
        sys.argv = argv

    text = (docs / "results.md").read_text()
    assert "## RQ1" in text and "0.900 ± 0.010" in text
    assert "_Not run: `results/summary/robustness.json` is missing._" in text
    assert (docs / "figures" / "rq1_quantum_head.png").stat().st_size > 1000
