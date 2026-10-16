"""Regression tests for FL findings F1-F5 and F8-F10 of the 99a5be6 audit."""

import importlib.util
import subprocess
import sys

import numpy as np
import pytest

from qedge.fl.compression import TopKCompressor
from qedge.fl.strategies import SERVER_STRATEGIES, ServerStrategy, fedadam

torch = pytest.importorskip("torch")


def _w(*values):
    return {"w": np.asarray(values, np.float32)}


def test_identical_updates_aggregate_to_themselves():
    w = _w(*np.linspace(-1.0, 1.0, 20))
    for name in SERVER_STRATEGIES:
        result = ServerStrategy(name).step(w, [(w, 1), (w, 2)])
        np.testing.assert_allclose(result["w"], w["w"], atol=1e-6)


def test_aggregation_never_zeroes_weights():
    ones = _w(*np.ones(100))
    for name in SERVER_STRATEGIES:
        assert np.all(ServerStrategy(name).step(ones, [(ones, 1)])["w"] != 0)


def test_aggregation_does_not_evaluate_quantum_kernels():
    code = "import sys, qedge.fl.strategies, qedge.fl.simulation; print('pennylane' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "False"


def test_fedprox_is_a_client_side_proximal_term():
    from qedge.fl.models import build_model, get_weights
    from qedge.fl.training import TrainConfig, local_train

    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 5)).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.int64)
    distances = {}
    for mu in (0.0, 10.0):
        model = build_model("logreg", 5, 2, seed=0)
        start = get_weights(model)
        local_train(model, x, y, TrainConfig(epochs=5, lr=0.1, mu=mu), seed=1)
        end = get_weights(model)
        distances[mu] = sum(float(np.sum((end[n] - start[n]) ** 2)) for n in start)

    assert distances[10.0] < 0.5 * distances[0.0]
    assert "fedprox" not in SERVER_STRATEGIES


def test_fedadam_keeps_optimizer_state_across_rounds():
    zero = _w(0.0, 0.0, 0.0)
    first, state = fedadam(zero, [(_w(1.0, 1.0, 1.0), 1)], None, lr=0.1)
    second, state = fedadam(first, [({"w": first["w"] + 1.0}, 1)], state, lr=0.1)

    m1 = 0.1 * 1.0
    v1 = 0.99 * 1e-6 + 0.01 * 1.0
    np.testing.assert_allclose(first["w"], 0.1 * m1 / (np.sqrt(v1) + 1e-3), rtol=1e-5)
    m2 = 0.9 * m1 + 0.1 * 1.0
    v2 = 0.99 * v1 + 0.01 * 1.0
    np.testing.assert_allclose(second["w"] - first["w"], 0.1 * m2 / (np.sqrt(v2) + 1e-3), rtol=1e-5)
    assert np.all(second["w"] - first["w"] > first["w"])


def test_quantum_parameters_are_not_trained_with_random_noise():
    assert importlib.util.find_spec("qedge.fl.aggregator") is None
    assert importlib.util.find_spec("qedge.fl.client") is None


def test_compression_applies_to_update_deltas():
    compressor = TopKCompressor(0.1)
    delta = {"w": np.zeros(50, np.float32)}
    weights = {"w": np.arange(50, dtype=np.float32)}

    sent, _ = compressor.compress(delta)

    np.testing.assert_array_equal(weights["w"] + sent["w"], weights["w"])


@pytest.mark.parametrize("module", ["qedge.fl.models", "qedge.fl.training", "qedge.fl.simulation"])
def test_fl_modules_report_missing_torch_as_import_error(module):
    code = f"import sys; sys.modules['torch'] = None; import {module}"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert "ImportError" in result.stderr and "q-edge[fl]" in result.stderr


def test_simulation_learns_real_labels():
    pytest.importorskip("sklearn")
    from qedge.fl.simulation import SimulationConfig, run_simulation

    config = SimulationConfig(dataset="breast_cancer", clients=5, rounds=8, alpha=0.5)

    assert run_simulation(config)["final_accuracy"] > 0.9
