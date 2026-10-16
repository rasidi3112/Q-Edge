"""Every configurable option either has its own implementation or raises."""

import numpy as np
import pennylane as qml
import pytest

from qedge.fl.strategies import SERVER_STRATEGIES, ServerStrategy
from qedge.quantum.circuits import ANSATZE, vqc, weight_shape
from qedge.quantum.kernels import FEATURE_MAPS, QuantumKernel
from qedge.quantum.noise import CHANNELS, NoiseConfig, make_device
from qedge.quantum.zne import METHODS, TensoredReadoutMitigation, ZNEConfig, extrapolate


def _assert_pairwise_distinct(outputs):
    members = list(outputs)
    for i, first in enumerate(members):
        for second in members[i + 1 :]:
            a, b = np.asarray(outputs[first]), np.asarray(outputs[second])
            same = a.shape == b.shape and np.allclose(a, b)
            label = [getattr(m, "name", m) for m in (first, second)]
            assert not same, f"{label[0]} and {label[1]} compute the same result"


def _vqc_output(ansatz, noise=None):
    weights = np.random.default_rng(7).uniform(0, 2 * np.pi, weight_shape(ansatz, 2, 3))
    qnode = qml.QNode(
        lambda x, w: vqc(x, w, ansatz=ansatz, reupload=False),
        make_device(3, noise or NoiseConfig()),
    )
    return qnode(np.array([0.1, 0.2, 0.3]), weights)


def test_every_ansatz_builds_a_distinct_circuit():
    _assert_pairwise_distinct({ansatz: _vqc_output(ansatz) for ansatz in ANSATZE})


def test_every_noise_channel_has_its_own_effect():
    _assert_pairwise_distinct(
        {
            channel: _vqc_output("strongly_entangling", NoiseConfig(p=0.05, channel=channel))
            for channel in CHANNELS
        }
    )


def test_every_feature_map_builds_a_distinct_kernel():
    x1, x2 = np.array([0.3, -0.7, 1.1]), np.array([-0.2, 0.9, 0.4])
    _assert_pairwise_distinct({kind: QuantumKernel(3, kind=kind)(x1, x2) for kind in FEATURE_MAPS})


def test_every_zne_method_has_its_own_fit():
    scales = (1.0, 2.0, 3.0, 4.0)
    values = [0.9, 0.7, 0.62, 0.45]
    _assert_pairwise_distinct(
        {
            method: extrapolate(scales, values, ZNEConfig(method, scales, order=2))
            for method in METHODS
        }
    )


def test_every_readout_mitigation_method_has_its_own_rule():
    device = make_device(2, NoiseConfig(readout=0.1))
    measured = np.array([0.05, 0.6, 0.3, 0.05])
    outputs = {}
    for method in ("inverse", "least_squares"):
        mitigation = TensoredReadoutMitigation(2, method)
        mitigation.calibrate(device)
        outputs[method] = mitigation.mitigate(measured)
    _assert_pairwise_distinct(outputs)


def test_every_aggregation_strategy_has_its_own_rule():
    rng = np.random.default_rng(0)
    rounds = [[({"w": rng.normal(size=6)}, 10 + i) for i in range(3)] for _ in range(3)]
    outputs = {}
    for name in SERVER_STRATEGIES:
        strategy = ServerStrategy(name)
        weights = {"w": np.zeros(6)}
        for updates in rounds:
            weights = strategy.step(weights, updates)
        outputs[name] = weights["w"]
    _assert_pairwise_distinct(outputs)


def test_unknown_options_raise():
    with pytest.raises(ValueError):
        weight_shape("brick", 1, 2)
    with pytest.raises(ValueError):
        QuantumKernel(2, kind="pauli")
    with pytest.raises(ValueError):
        ZNEConfig(method="adaptive")
    with pytest.raises(ValueError):
        NoiseConfig(channel="phase_flip")
    with pytest.raises(ValueError):
        ServerStrategy("scaffold")
