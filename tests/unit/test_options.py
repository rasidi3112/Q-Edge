"""Every configurable option either has its own implementation or raises."""

import numpy as np
import pytest

from qedge.fl.aggregator import (
    AggregationStrategy,
    LocalModelUpdate,
    QuantumAggregatorConfig,
    QuantumGlobalAggregator,
)
from qedge.quantum.circuits import AnsatzType, VariationalQuantumCircuit, VQCConfig
from qedge.quantum.kernels import FeatureMapType, QKAConfig, QuantumKernelAlignment
from qedge.quantum.zne import MeasurementErrorMitigation

ALIASED = pytest.mark.xfail(strict=True, reason="some enum members fall back to another member")
UNVALIDATED = pytest.mark.xfail(strict=True, reason="unknown option falls back to a default")


def _assert_pairwise_distinct(outputs):
    members = list(outputs)
    for i, first in enumerate(members):
        for second in members[i + 1 :]:
            a, b = np.asarray(outputs[first]), np.asarray(outputs[second])
            same = a.shape == b.shape and np.allclose(a, b)
            assert not same, f"{first.name} and {second.name} compute the same result"


@ALIASED
def test_every_ansatz_type_builds_a_distinct_circuit():
    features = np.array([0.1, 0.2, 0.3, 0.4])
    outputs = {
        ansatz: VariationalQuantumCircuit(
            VQCConfig(n_qubits=4, n_layers=2, ansatz_type=ansatz), seed=7
        ).forward(features)
        for ansatz in AnsatzType
    }
    _assert_pairwise_distinct(outputs)


@ALIASED
def test_every_feature_map_type_builds_a_distinct_kernel():
    x1, x2 = np.array([0.3, -0.7, 1.1]), np.array([-0.2, 0.9, 0.4])
    outputs = {
        fmap: QuantumKernelAlignment(
            QKAConfig(n_qubits=3, n_layers=1, feature_map_type=fmap), seed=7
        ).evaluate_kernel(x1, x2)
        for fmap in FeatureMapType
    }
    _assert_pairwise_distinct(outputs)


@ALIASED
@pytest.mark.asyncio
async def test_every_aggregation_strategy_has_its_own_rule():
    rng = np.random.default_rng(0)
    rounds = [
        [LocalModelUpdate(f"c{i}", rng.normal(size=6), n_samples=10 + i) for i in range(3)]
        for _ in range(2)
    ]
    outputs = {}
    for strategy in AggregationStrategy:
        aggregator = QuantumGlobalAggregator(
            QuantumAggregatorConfig(n_qubits=2, vqc_layers=1, aggregation_strategy=strategy),
            seed=0,
        )
        for updates in rounds:
            state = await aggregator.aggregate(updates, use_quantum=False)
        outputs[strategy] = state.weights
    _assert_pairwise_distinct(outputs)


@UNVALIDATED
def test_measurement_error_mitigation_rejects_unknown_method():
    with pytest.raises(ValueError, match="method"):
        MeasurementErrorMitigation(n_qubits=2, method="bogus")


@UNVALIDATED
def test_align_to_target_rejects_unknown_target_kernel():
    qka = QuantumKernelAlignment(QKAConfig(n_qubits=2, n_layers=1, max_iterations=1), seed=0)
    X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    y = np.array([0, 1, 0])
    with pytest.raises(ValueError, match="kernel_type"):
        qka.align_to_target(X, y, kernel_type="bogus")


@UNVALIDATED
def test_draw_rejects_unknown_style():
    vqc = VariationalQuantumCircuit(VQCConfig(n_qubits=2, n_layers=1), seed=0)
    with pytest.raises(ValueError, match="style"):
        vqc.draw(style="bogus")


@UNVALIDATED
def test_hardware_efficient_ansatz_sizes_parameters_from_rotation_gates():
    config = VQCConfig(
        n_qubits=3,
        n_layers=2,
        ansatz_type=AnsatzType.HARDWARE_EFFICIENT,
        rotation_gates=["RY", "Rot"],
    )
    vqc = VariationalQuantumCircuit(config, seed=0)

    probs = vqc.forward(np.array([0.1, 0.2, 0.3]))

    assert vqc.num_params == 2 * 3 * 4
    assert np.isclose(np.sum(probs), 1.0)
