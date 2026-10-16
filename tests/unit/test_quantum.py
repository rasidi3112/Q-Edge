import numpy as np
import pennylane as qml
import pytest

from qedge.quantum.circuits import ANSATZE, probabilities_circuit, vqc, weight_shape
from qedge.quantum.kernels import QuantumKernel, parameter_shape, target_alignment
from qedge.quantum.noise import NoiseConfig, make_device
from qedge.quantum.zne import (
    TensoredReadoutMitigation,
    ZNEConfig,
    achieved_scale_factors,
    extrapolate,
    mitigate,
)


def _qnode(device, ansatz="strongly_entangling", reupload=False):
    return qml.QNode(lambda x, w: vqc(x, w, ansatz=ansatz, reupload=reupload), device)


@pytest.mark.parametrize("ansatz", ANSATZE)
def test_vqc_returns_bounded_expectations_per_qubit(ansatz):
    weights = np.random.default_rng(0).uniform(0, 2 * np.pi, weight_shape(ansatz, 2, 3))
    out = np.asarray(
        _qnode(make_device(3, NoiseConfig()), ansatz)(np.array([0.1, 0.2, 0.3]), weights)
    )

    assert out.shape == (3,)
    assert np.all(np.abs(out) <= 1 + 1e-9)


def test_reuploading_changes_the_circuit():
    weights = np.random.default_rng(0).uniform(
        0, 2 * np.pi, weight_shape("strongly_entangling", 2, 2)
    )
    x = np.array([0.4, -0.9])
    device = make_device(2, NoiseConfig())

    plain = np.asarray(_qnode(device)(x, weights))
    reuploaded = np.asarray(_qnode(device, reupload=True)(x, weights))

    assert not np.allclose(plain, reuploaded)


def test_weight_shape_rejects_unknown_ansatz():
    with pytest.raises(ValueError, match="Unknown ansatz"):
        weight_shape("brick", 1, 2)


def test_probabilities_sum_to_one():
    device = make_device(3, NoiseConfig())
    weights = np.zeros(weight_shape("strongly_entangling", 1, 3))
    probs = qml.QNode(
        lambda x, w: probabilities_circuit(x, w, ansatz="strongly_entangling"), device
    )

    assert float(np.sum(probs(np.array([0.3, 0.0, 0.0]), weights))) == pytest.approx(1.0)


def test_depolarizing_noise_shrinks_expectations():
    weights = np.zeros(weight_shape("strongly_entangling", 1, 2))
    x = np.zeros(2)
    ideal = np.asarray(_qnode(make_device(2, NoiseConfig()))(x, weights))
    noisy = np.asarray(_qnode(make_device(2, NoiseConfig(p=0.05)))(x, weights))

    np.testing.assert_allclose(ideal, 1.0)
    assert np.all(noisy < 0.95)


@pytest.mark.parametrize(
    "kwargs", [{"p": 1.0}, {"readout": -0.1}, {"channel": "phase"}, {"shots": 0}]
)
def test_noise_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        NoiseConfig(**kwargs)


def test_shots_give_sampled_estimates():
    device = make_device(1, NoiseConfig(), seed=0)

    @qml.qnode(device)
    def circuit():
        qml.RY(np.pi / 2, wires=0)
        return qml.expval(qml.PauliZ(0))

    sampled = qml.set_shots(circuit, shots=200)()
    # The exact value is 0. Unseeded, 200 shots give exactly 0 about 6% of the time.
    assert abs(float(sampled)) < 0.3 and float(sampled) != 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"method": "linear"}, "Unknown ZNE method"),
        ({"scale_factors": (2.0, 3.0)}, "start at 1"),
        ({"scale_factors": (1.0, 1.0)}, "increase"),
        ({"method": "poly", "order": 2, "scale_factors": (1.0, 3.0)}, "order"),
    ],
)
def test_zne_config_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ZNEConfig(**kwargs)


@pytest.mark.parametrize(
    ("config", "values", "expected"),
    [
        (ZNEConfig(scale_factors=(1.0, 3.0)), [0.8, 0.4], 1.0),
        (ZNEConfig("poly", (1.0, 2.0, 3.0), order=2), [0.9, 0.6, 0.1], 1.0),
        (ZNEConfig("exponential", (1.0, 2.0, 3.0)), [np.exp(-0.2 * s) for s in (1, 2, 3)], 1.0),
    ],
)
def test_extrapolators_recover_known_models(config, values, expected):
    assert extrapolate(config.scale_factors, values, config) == pytest.approx(expected, abs=1e-9)


def test_partial_folding_reports_the_factor_it_reaches():
    tape = qml.tape.QuantumScript([qml.RX(0.1, 0), qml.RY(0.2, 0), qml.RZ(0.3, 0)])

    assert achieved_scale_factors(tape, (1.0, 1.5, 3.0)) == [1.0, 5 / 3, 3.0]


def test_mitigated_qnode_is_differentiable_and_needs_odd_factors():
    torch = pytest.importorskip("torch")
    device = make_device(1, NoiseConfig(p=0.02))

    def circuit(theta):
        qml.RX(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    qnode = qml.QNode(circuit, device, interface="torch", diff_method="backprop")
    theta = torch.tensor(0.3, requires_grad=True)
    value = mitigate(qnode, ZNEConfig())(theta)
    value.backward()

    assert float(value.detach()) == pytest.approx(np.cos(0.3), abs=2e-3)
    assert float(theta.grad) == pytest.approx(-np.sin(0.3), abs=1e-2)
    with pytest.raises(ValueError, match="odd integer"):
        mitigate(qnode, ZNEConfig(scale_factors=(1.0, 2.0)))


@pytest.mark.parametrize("method", ["inverse", "least_squares"])
def test_tensored_readout_mitigation_recovers_the_ideal_distribution(method):
    device = make_device(3, NoiseConfig(readout=0.05))
    mitigation = TensoredReadoutMitigation(3, method)
    mitigation.calibrate(device)

    def circuit():
        qml.PauliX(wires=0)
        qml.Hadamard(wires=1)
        return qml.probs(wires=range(3))

    measured = qml.QNode(circuit, device)()
    ideal = qml.QNode(circuit, make_device(3, NoiseConfig()))()

    assert mitigation.calibration_circuits == 6
    np.testing.assert_allclose(mitigation.confusion[0], [[0.95, 0.05], [0.05, 0.95]], atol=1e-9)
    assert np.abs(measured - ideal).max() > 0.04
    np.testing.assert_allclose(mitigation.mitigate(measured), ideal, atol=1e-4)


def test_readout_mitigation_requires_calibration_and_a_known_method():
    with pytest.raises(ValueError, match="method"):
        TensoredReadoutMitigation(2, "iterative")
    with pytest.raises(RuntimeError, match="calibrate"):
        TensoredReadoutMitigation(2).mitigate(np.full(4, 0.25))


def test_kernel_rejects_bad_shapes_and_feature_maps():
    with pytest.raises(ValueError, match="Unknown feature map"):
        QuantumKernel(2, kind="pauli")
    with pytest.raises(ValueError, match="shape"):
        QuantumKernel(2, params=np.ones((3, 3)))
    with pytest.raises(ValueError, match="features"):
        QuantumKernel(2)(np.zeros(3), np.zeros(3))


def test_feature_maps_differ_and_matrix_is_symmetric():
    x = np.random.default_rng(0).normal(size=(5, 3))
    angle, zz = QuantumKernel(3, kind="angle").matrix(x), QuantumKernel(3, kind="zz").matrix(x)

    assert not np.allclose(angle, zz)
    np.testing.assert_allclose(angle, angle.T)
    np.testing.assert_allclose(np.diag(angle), 1.0)
    assert QuantumKernel(3).matrix(x[:2], x).shape == (2, 5)
    assert parameter_shape("zz", 2, 3) == (2, 5)


def test_target_alignment_is_one_for_the_ideal_kernel():
    y = np.array([1, 1, -1, -1])
    assert target_alignment(np.outer(y, y), y) == pytest.approx(1.0)
