"""Regression tests for FL findings F1-F5 and F8-F10 of the 99a5be6 audit."""

import subprocess
import sys

import numpy as np
import pytest

AUDIT = pytest.mark.xfail(strict=True, reason="FL audit finding")


def _aggregator(strategy="fedavg"):
    from qedge.fl.aggregator import (
        AggregationStrategy,
        QuantumAggregatorConfig,
        QuantumGlobalAggregator,
    )

    config = QuantumAggregatorConfig(
        n_qubits=4,
        vqc_layers=1,
        use_error_mitigation=False,
        aggregation_strategy=AggregationStrategy(strategy),
    )
    return QuantumGlobalAggregator(config, seed=0)


def _update(client_id, weights, n):
    from qedge.fl.aggregator import LocalModelUpdate

    return LocalModelUpdate(client_id, np.asarray(weights, np.float64), n_samples=n)


@AUDIT
@pytest.mark.asyncio
async def test_identical_updates_aggregate_to_themselves():
    w = np.linspace(-1.0, 1.0, 20)
    state = await _aggregator().aggregate(
        [_update("a", w, 1), _update("b", w, 2)], use_quantum=True
    )

    np.testing.assert_allclose(state.weights, w, atol=1e-9)


@AUDIT
def test_aggregation_never_zeroes_weights():
    expanded = _aggregator()._expand_weights(np.ones(8), (100,))

    assert np.all(expanded != 0)


@AUDIT
@pytest.mark.asyncio
async def test_aggregation_does_not_evaluate_quantum_kernels(monkeypatch):
    aggregator = _aggregator()

    def forbidden(*args, **kwargs):
        raise AssertionError("aggregation evaluated a quantum kernel")

    monkeypatch.setattr(aggregator.qka, "compute_kernel_matrix", forbidden)
    w = np.ones(10)
    await aggregator.aggregate([_update("a", w, 1), _update("b", w, 1)], use_quantum=True)


@AUDIT
@pytest.mark.asyncio
async def test_fedprox_is_a_client_side_proximal_term():
    aggregator = _aggregator("fedprox")
    await aggregator.aggregate([_update("a", np.zeros(4), 1)], use_quantum=False)
    w = np.ones(4)
    state = await aggregator.aggregate([_update("a", w, 1), _update("b", w, 1)], use_quantum=False)

    np.testing.assert_allclose(state.weights, w)


@AUDIT
@pytest.mark.asyncio
async def test_fedadam_keeps_optimizer_state_across_rounds():
    aggregator = _aggregator("fedopt")
    await aggregator.aggregate([_update("a", np.zeros(3), 1)], use_quantum=False)
    first = (await aggregator.aggregate([_update("a", np.ones(3), 1)], use_quantum=False)).weights
    target = first + 1.0
    second = (await aggregator.aggregate([_update("a", target, 1)], use_quantum=False)).weights

    # Adam's first moment accumulates, so the second step is larger than a stateless one.
    assert np.all(second - first > 0.01 * (target - first) + 1e-12)


@AUDIT
def test_quantum_parameters_are_not_trained_with_random_noise():
    import importlib.util

    assert importlib.util.find_spec("qedge.fl.aggregator") is None


@AUDIT
def test_compression_applies_to_update_deltas():
    pytest.importorskip("torch")
    from qedge.fl.client import MobileDeviceConfig, MobileFlowerClient

    client = MobileFlowerClient(MobileDeviceConfig(compression_ratio=0.1))
    weights = client.get_parameters()

    sent = client.compress_gradients(weights)

    for original, transmitted in zip(weights, sent, strict=True):
        np.testing.assert_allclose(transmitted, original)


@AUDIT
def test_fl_modules_report_missing_torch_as_import_error():
    code = "import sys; sys.modules['torch'] = None; import qedge.fl.client"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert "ImportError" in result.stderr and "q-edge[fl]" in result.stderr


@AUDIT
def test_simulation_learns_real_labels():
    from qedge.fl.simulation import SimulationConfig, run_simulation

    config = SimulationConfig(dataset="breast_cancer", clients=5, rounds=8, alpha=0.5)

    assert run_simulation(config)["final_accuracy"] > 0.9
