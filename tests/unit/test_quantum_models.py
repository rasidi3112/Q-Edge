import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

import torch

from qedge.cli import main
from qedge.fl.models import build_model, count_parameters, get_weights
from qedge.fl.simulation import SimulationConfig, run_simulation
from qedge.quantum.models import (
    QuantumHeadConfig,
    build_head_model,
    head_config,
    matched_head_shape,
)
from qedge.quantum.noise import NoiseConfig
from qedge.quantum.zne import ZNEConfig


def _model(name="hybrid-vqc", seed=0, **options):
    return build_head_model(name, 5, 3, head_config(**options), seed=seed)


@pytest.mark.parametrize(("qubits", "layers"), [(2, 1), (4, 2), (4, 3), (6, 2)])
def test_classical_head_matches_the_vqc_parameter_count(qubits, layers):
    hybrid = _model(n_qubits=qubits, n_layers=layers)
    classical = _model("classical-head", n_qubits=qubits, n_layers=layers)
    vqc = qubits * layers * 3

    assert count_parameters(hybrid.head) == vqc
    assert abs(count_parameters(classical.head) - vqc) <= qubits + 1
    assert count_parameters(hybrid.encoder) == count_parameters(classical.encoder)


def test_classical_head_is_never_narrower_than_the_qubit_count():
    # 4 qubits, 2 layers: VQC 24; Linear(4, 4) has 20, a hidden width 4 has 40.
    assert matched_head_shape(4, 24) is None
    assert matched_head_shape(4, 36) == 4
    assert matched_head_shape(4, 100) == 11
    for n, target in ((2, 6), (4, 0), (6, 36), (8, 96)):
        hidden = matched_head_shape(n, target)
        assert hidden is None or hidden >= n


@pytest.mark.parametrize("name", ["hybrid-vqc", "classical-head"])
def test_head_models_output_logits_and_train(name):
    model = _model(name)
    x = torch.randn(7, 5)

    logits = model(x)
    torch.nn.functional.cross_entropy(logits, torch.zeros(7, dtype=torch.long)).backward()

    assert logits.shape == (7, 3)
    assert all(p.grad is not None and torch.any(p.grad != 0) for p in model.head.parameters())


def test_initialization_is_seeded_and_leaves_the_global_rng_alone():
    torch.manual_seed(123)
    before = torch.rand(1)
    torch.manual_seed(123)
    a, b, c = _model(seed=1), _model(seed=1), _model(seed=2)
    after = torch.rand(1)

    wa, wb, wc = (get_weights(m) for m in (a, b, c))
    assert all(np.array_equal(wa[n], wb[n]) for n in wa)
    assert not np.array_equal(wa["head.weights"], wc["head.weights"])
    assert torch.equal(before, after)


def test_noise_changes_the_output_and_zne_records_its_factors():
    x = torch.randn(4, 5)
    clean = _model(seed=0)
    noisy = _model(seed=0, noise=0.05)
    mitigated = _model(seed=0, noise=0.05, zne="richardson")

    assert not torch.allclose(clean(x), noisy(x))
    assert mitigated.describe()["zne_achieved_scale_factors"] == [1.0, 3.0, 5.0]
    assert noisy.describe()["zne_achieved_scale_factors"] is None


def test_mitigated_head_is_closer_to_the_noiseless_head():
    angles = torch.rand(6, 4) * 3
    clean = _model(seed=0).head(angles)
    noisy = _model(seed=0, noise=0.02).head(angles)
    mitigated = _model(seed=0, noise=0.02, zne="richardson").head(angles)

    assert (mitigated - clean).abs().mean() < 0.3 * (noisy - clean).abs().mean()


def test_shots_switch_the_vqc_to_parameter_shift():
    model = _model(n_qubits=2, n_layers=1, shots=50)

    assert model.head.qnode.diff_method == "parameter-shift"
    assert _model(n_qubits=2, n_layers=1).head.qnode.diff_method == "backprop"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ansatz": "brick"},
        {"n_qubits": 0},
        {"zne": ZNEConfig()},
    ],
)
def test_head_config_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        QuantumHeadConfig(**kwargs, noise=NoiseConfig())


def test_fl_build_model_validates_options():
    with pytest.raises(ValueError, match="do not apply"):
        build_model("logreg", 4, 2, seed=0, hidden=3)
    with pytest.raises(ValueError, match="Invalid options"):
        build_model("hybrid-vqc", 4, 2, seed=0, qubits=3)
    assert count_parameters(build_model("mlp", 4, 2, seed=0, hidden=3)) == 4 * 3 + 3 + 3 * 2 + 2


def test_simulation_records_head_size_and_learns(tmp_path):
    config = SimulationConfig(
        model="hybrid-vqc", clients=4, rounds=12, model_options={"n_qubits": 3, "n_layers": 1}
    )

    summary = run_simulation(config, tmp_path)

    assert summary["model"] == {
        "head_parameters": 9,
        "vqc_parameters": 9,
        "zne_achieved_scale_factors": None,
    }
    assert summary["final_accuracy"] > 0.85
    assert "head_parameters: 9" in next(tmp_path.iterdir()).joinpath("config.yaml").read_text()


def test_cli_passes_head_options_and_rejects_them_for_other_models(tmp_path, capsys):
    base = ["simulate", "--clients", "3", "--rounds", "1", "--out", str(tmp_path)]

    assert main([*base, "--model", "hybrid-vqc", "--qubits", "2", "--layers", "1"]) == 0
    assert "head params:    6" in capsys.readouterr().out
    assert main([*base, "--noise", "0.01"]) == 1
    assert "only apply" in capsys.readouterr().err


def test_quantum_models_report_missing_torch_as_import_error():
    code = "import sys; sys.modules['torch'] = None; import qedge.quantum.models"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert "ImportError" in result.stderr and "q-edge[fl]" in result.stderr


@pytest.mark.slow
def test_noisy_mitigated_hybrid_model_learns():
    options = {"n_qubits": 3, "n_layers": 1, "noise": 0.01, "zne": "richardson"}
    config = SimulationConfig(model="hybrid-vqc", clients=4, rounds=10, model_options=options)

    summary = run_simulation(config)

    assert summary["model"]["zne_achieved_scale_factors"] == [1.0, 3.0, 5.0]
    assert summary["final_accuracy"] > 0.85
