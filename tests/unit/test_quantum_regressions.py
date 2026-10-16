"""Regression tests for quantum findings Q1-Q6 of the 99a5be6 audit."""

import numpy as np
import pennylane as qml
import pytest

from qedge.quantum.kernels import QuantumKernel, parameter_shape
from qedge.quantum.noise import NoiseConfig, make_device
from qedge.quantum.zne import ZNEConfig, estimate, extrapolate


def _kernel(kind="angle", n_qubits=3, n_layers=2, seed=0):
    shape = parameter_shape(kind, n_layers, n_qubits)
    params = np.random.default_rng(seed).uniform(-np.pi, np.pi, shape)
    return QuantumKernel(n_qubits, n_layers, kind, params=params)


def _circuit(theta):
    qml.RX(theta, wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(0.3, wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


@pytest.mark.parametrize("kind", ["angle", "zz"])
def test_kernel_of_a_point_with_itself_is_one(kind):
    kernel = _kernel(kind)
    for x in np.random.default_rng(1).uniform(-np.pi, np.pi, (5, 3)):
        assert kernel(x, x) == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize("kind", ["angle", "zz"])
def test_kernel_matrix_is_positive_semidefinite(kind):
    x = np.random.default_rng(2).uniform(-np.pi, np.pi, (8, 3))

    assert np.linalg.eigvalsh(_kernel(kind).matrix(x)).min() >= -1e-8


def test_default_kernel_is_not_concentrated():
    x = np.random.default_rng(3).normal(size=(10, 4))
    matrix = QuantumKernel(4).matrix(x)

    assert matrix[np.triu_indices(10, 1)].std() > 0.05


def test_noise_scaled_circuits_return_finite_values():
    noisy = qml.QNode(_circuit, make_device(2, NoiseConfig(p=0.01)))

    result = estimate(noisy, (0.4,), ZNEConfig(scale_factors=(1.0, 2.0, 3.0)))

    assert np.all(np.isfinite(result.values))
    assert result.achieved == (1.0, 2.0, 3.0)


def test_extrapolation_is_not_chosen_by_in_sample_fit():
    scales = (1.0, 2.0, 3.0)
    values = 0.5 + 0.5 * np.exp(-0.3 * np.asarray(scales))  # true zero-noise value 1.0

    with pytest.raises(ValueError, match="Unknown ZNE method"):
        ZNEConfig(method="adaptive")
    estimate_ = extrapolate(scales, values, ZNEConfig(scale_factors=scales))
    assert estimate_ == pytest.approx(1.0, abs=0.01)


def test_improvement_is_measured_against_the_unmitigated_value():
    noisy = qml.QNode(_circuit, make_device(2, NoiseConfig(p=0.01)))

    result = estimate(noisy, (0.4,), ZNEConfig(scale_factors=(1.0, 3.0, 5.0)))

    assert result.unmitigated == result.values[0] == pytest.approx(float(noisy(0.4)))


def test_zne_runs_on_a_noisy_device():
    ideal = float(qml.QNode(_circuit, make_device(2, NoiseConfig()))(0.4))
    noisy = qml.QNode(_circuit, make_device(2, NoiseConfig(p=0.01)))

    result = estimate(noisy, (0.4,), ZNEConfig(scale_factors=(1.0, 3.0, 5.0)))

    assert noisy.device.name == "default.mixed"
    assert abs(result.unmitigated - ideal) > 0.02
    assert abs(result.mitigated - ideal) < 0.25 * abs(result.unmitigated - ideal)
