"""Regression tests for quantum findings Q1-Q6 of the 99a5be6 audit."""

import numpy as np
import pytest

AUDIT = pytest.mark.xfail(strict=True, reason="quantum audit finding")


def _kernel(n_qubits=3, n_layers=2, scale=None, seed=0):
    from qedge.quantum.kernels import QKAConfig, QuantumKernelAlignment

    kernel = QuantumKernelAlignment(QKAConfig(n_qubits=n_qubits, n_layers=n_layers), seed=seed)
    if scale is not None:
        shape = kernel.feature_map_params.shape
        kernel.feature_map_params = np.random.default_rng(seed).uniform(-scale, scale, shape)
    return kernel


@AUDIT
def test_kernel_of_a_point_with_itself_is_one():
    kernel = _kernel(scale=np.pi)
    for x in np.random.default_rng(1).uniform(-np.pi, np.pi, (5, 3)):
        assert kernel.evaluate_kernel(x, x) == pytest.approx(1.0, abs=1e-8)


@AUDIT
def test_kernel_matrix_is_positive_semidefinite():
    kernel = _kernel(scale=np.pi)
    x = np.random.default_rng(2).uniform(-np.pi, np.pi, (8, 3))

    assert np.linalg.eigvalsh(kernel.compute_kernel_matrix(x)).min() >= -1e-8


@AUDIT
def test_default_kernel_is_not_concentrated():
    kernel = _kernel(n_qubits=4)
    x = np.random.default_rng(3).normal(size=(10, 4))
    matrix = kernel.compute_kernel_matrix(x)

    assert matrix[np.triu_indices(10, 1)].std() > 0.05


@AUDIT
def test_noise_scaled_circuits_return_finite_values():
    import pennylane as qml

    from qedge.quantum.zne import ZeroNoiseExtrapolation, ZNEConfig

    device = qml.device("default.qubit", wires=2)

    @qml.qnode(device)
    def circuit(theta):
        qml.RX(theta, wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    zne = ZeroNoiseExtrapolation(ZNEConfig(scale_factors=[1.0, 2.0, 3.0]))
    values = [zne.execute_at_scale(circuit, 0.4, s, device) for s in (1.0, 2.0, 3.0)]

    assert np.all(np.isfinite(values))


@AUDIT
def test_extrapolation_is_not_chosen_by_in_sample_fit():
    from qedge.quantum.zne import ExtrapolationMethod, ZeroNoiseExtrapolation, ZNEConfig

    scales = np.array([1.0, 2.0, 3.0])
    values = 0.5 + 0.5 * np.exp(-0.3 * scales)  # true zero-noise value 1.0
    zne = ZeroNoiseExtrapolation(
        ZNEConfig(scale_factors=scales, extrapolation_method=ExtrapolationMethod.ADAPTIVE)
    )

    assert zne.extrapolate(scales, values)["mitigated_value"] == pytest.approx(1.0, abs=1e-3)


@AUDIT
def test_improvement_is_measured_against_the_unmitigated_value(monkeypatch):
    from qedge.quantum.zne import ExtrapolationMethod, ZeroNoiseExtrapolation, ZNEConfig

    zne = ZeroNoiseExtrapolation(
        ZNEConfig(scale_factors=[1.0, 2.0, 3.0], extrapolation_method=ExtrapolationMethod.LINEAR)
    )
    measured = {1.0: 0.8, 2.0: 0.6, 3.0: 0.4}
    monkeypatch.setattr(zne, "execute_at_scale", lambda c, p, scale, d: measured[scale])

    result = zne.mitigate(lambda p: 0.0, None, device=object())

    assert result["mitigated_value"] == pytest.approx(1.0)
    assert result["improvement"] == pytest.approx(abs(1.0 - 0.8))


@AUDIT
def test_zne_runs_on_a_noisy_device():
    from qedge.quantum.noise import NoiseConfig, make_device

    assert make_device(2, NoiseConfig(p=0.01)).name == "default.mixed"
