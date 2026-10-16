import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

# Import modules to test
from qedge.quantum.circuits import (
    VariationalQuantumCircuit,
    VQCConfig,
    EntanglementPattern,
    AnsatzType,
)
from qedge.quantum.kernels import (
    QuantumKernelAlignment,
    QKAConfig,
    FeatureMapType,
)
from qedge.quantum.zne import (
    ZeroNoiseExtrapolation,
    ZNEConfig,
    ExtrapolationMethod,
    MeasurementErrorMitigation,
)


class TestVariationalQuantumCircuit:
    def test_vqc_initialization(self):
        config = VQCConfig(n_qubits=4, n_layers=2)
        vqc = VariationalQuantumCircuit(config)

        assert vqc.config.n_qubits == 4
        assert vqc.config.n_layers == 2
        assert vqc.params is not None
        assert vqc.num_params > 0

    def test_vqc_forward_pass(self):
        config = VQCConfig(n_qubits=4, n_layers=2)
        vqc = VariationalQuantumCircuit(config)

        features = np.random.randn(4)
        output = vqc.forward(features)

        # Should return probabilities for 2^n states
        assert output.shape == (16,)
        assert np.isclose(np.sum(output), 1.0, atol=1e-6)
        assert np.all(output >= 0)

    def test_vqc_batch_forward(self):
        config = VQCConfig(n_qubits=4, n_layers=2)
        vqc = VariationalQuantumCircuit(config)

        batch = np.random.randn(5, 4)
        outputs = vqc.batch_forward(batch)

        assert outputs.shape == (5, 16)
        for output in outputs:
            assert np.isclose(np.sum(output), 1.0, atol=1e-6)

    def test_vqc_different_entanglement_patterns(self):
        for pattern in EntanglementPattern:
            config = VQCConfig(n_qubits=4, n_layers=2, entanglement=pattern)
            vqc = VariationalQuantumCircuit(config)

            features = np.zeros(4)
            output = vqc.forward(features)

            assert output.shape == (16,)

    def test_vqc_serialization(self):
        config = VQCConfig(n_qubits=4, n_layers=2)
        vqc = VariationalQuantumCircuit(config, seed=42)

        # Serialize
        data = vqc.to_dict()

        # Deserialize
        vqc_restored = VariationalQuantumCircuit.from_dict(data)

        assert vqc_restored.config.n_qubits == vqc.config.n_qubits
        assert_array_almost_equal(vqc_restored.params, vqc.params)

    def test_vqc_gate_count(self):
        config = VQCConfig(
            n_qubits=4,
            n_layers=2,
            entanglement=EntanglementPattern.LINEAR,
        )
        vqc = VariationalQuantumCircuit(config)

        gate_counts = vqc.get_gate_count()

        assert "CNOT" in gate_counts
        assert "Rot" in gate_counts or "RY" in gate_counts
        assert all(count > 0 for count in gate_counts.values())

    def test_vqc_reproducibility(self):
        config = VQCConfig(n_qubits=4, n_layers=2)

        vqc1 = VariationalQuantumCircuit(config, seed=42)
        vqc2 = VariationalQuantumCircuit(config, seed=42)

        features = np.array([0.1, 0.2, 0.3, 0.4])

        output1 = vqc1.forward(features)
        output2 = vqc2.forward(features)

        assert_array_almost_equal(output1, output2)


class TestQuantumKernelAlignment:
    def test_qka_initialization(self):
        config = QKAConfig(n_qubits=4, n_layers=2)
        qka = QuantumKernelAlignment(config)

        assert qka.config.n_qubits == 4
        assert qka.feature_map_params is not None

    def test_kernel_evaluation(self):
        config = QKAConfig(n_qubits=4, n_layers=1)
        qka = QuantumKernelAlignment(config)

        x1 = np.random.randn(4)
        x2 = np.random.randn(4)

        kernel_value = qka.evaluate_kernel(x1, x2)

        # Kernel value should be in [0, 1]
        assert 0 <= kernel_value <= 1

    def test_kernel_self_similarity(self):
        config = QKAConfig(n_qubits=4, n_layers=1)
        qka = QuantumKernelAlignment(config)

        x = np.random.randn(4)

        kernel_value = qka.evaluate_kernel(x, x)

        # k(x, x) should be close to 1
        assert np.isclose(kernel_value, 1.0, atol=0.1)

    def test_kernel_matrix_computation(self):
        config = QKAConfig(n_qubits=4, n_layers=1)
        qka = QuantumKernelAlignment(config)

        X = np.random.randn(5, 4)
        K = qka.compute_kernel_matrix(X)

        # Kernel matrix should be symmetric
        assert K.shape == (5, 5)
        assert_array_almost_equal(K, K.T, decimal=5)

        # Diagonal should be close to 1
        assert np.allclose(np.diag(K), 1.0, atol=0.1)

    def test_kernel_alignment_score(self):
        config = QKAConfig(n_qubits=4, n_layers=1)
        qka = QuantumKernelAlignment(config)

        K1 = np.eye(5)
        K2 = np.eye(5)

        alignment = qka.kernel_alignment_score(K1, K2)

        # Same matrices should have alignment 1
        assert np.isclose(alignment, 1.0, atol=1e-6)

    def test_kernel_alignment_optimization(self):
        config = QKAConfig(
            n_qubits=4,
            n_layers=1,
            max_iterations=10,  # Reduced for testing
        )
        qka = QuantumKernelAlignment(config)

        X = np.random.randn(10, 4)
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        result = qka.align_to_target(X, y, kernel_type="ideal")

        assert "final_alignment" in result
        assert "n_iterations" in result
        assert result["final_alignment"] >= 0

    def test_qka_serialization(self):
        config = QKAConfig(n_qubits=4, n_layers=1)
        qka = QuantumKernelAlignment(config, seed=42)

        data = qka.to_dict()
        qka_restored = QuantumKernelAlignment.from_dict(data)

        assert qka_restored.config.n_qubits == qka.config.n_qubits
        assert_array_almost_equal(
            qka_restored.feature_map_params,
            qka.feature_map_params,
        )


class TestZeroNoiseExtrapolation:
    def test_zne_initialization(self):
        config = ZNEConfig(scale_factors=[1.0, 2.0, 3.0])
        zne = ZeroNoiseExtrapolation(config)

        assert len(zne.config.scale_factors) == 3

    def test_linear_extrapolation(self):
        config = ZNEConfig(
            scale_factors=[1.0, 2.0, 3.0],
            extrapolation_method=ExtrapolationMethod.LINEAR,
        )
        zne = ZeroNoiseExtrapolation(config)

        # Linear decay: f(λ) = 1.0 - 0.1λ
        scales = np.array([1.0, 2.0, 3.0])
        values = 1.0 - 0.1 * scales

        result = zne.extrapolate(scales, values)

        # Should extrapolate to ~1.0 at λ=0
        assert np.isclose(result["mitigated_value"], 1.0, atol=0.05)

    def test_polynomial_extrapolation(self):
        config = ZNEConfig(
            scale_factors=[1.0, 1.5, 2.0, 2.5],
            extrapolation_method=ExtrapolationMethod.POLYNOMIAL,
            poly_degree=2,
        )
        zne = ZeroNoiseExtrapolation(config)

        # Quadratic decay
        scales = np.array([1.0, 1.5, 2.0, 2.5])
        values = 1.0 - 0.1 * scales - 0.02 * scales**2

        result = zne.extrapolate(scales, values)

        assert "mitigated_value" in result
        assert result["method_used"] == "polynomial"

    def test_richardson_extrapolation(self):
        config = ZNEConfig(
            scale_factors=[1.0, 2.0, 3.0],
            extrapolation_method=ExtrapolationMethod.RICHARDSON,
        )
        zne = ZeroNoiseExtrapolation(config)

        scales = np.array([1.0, 2.0, 3.0])
        values = np.array([0.9, 0.8, 0.7])

        result = zne.extrapolate(scales, values)

        assert "mitigated_value" in result
        assert result["method_used"] == "richardson"

    def test_zne_config_validation(self):
        # Should raise for less than 2 scale factors
        with pytest.raises(ValueError):
            ZNEConfig(scale_factors=[1.0])

        # Should raise for scale factor < 1
        with pytest.raises(ValueError):
            ZNEConfig(scale_factors=[0.5, 1.0, 1.5])


class TestMeasurementErrorMitigation:
    def test_mem_initialization(self):
        mem = MeasurementErrorMitigation(n_qubits=2)

        assert mem.n_qubits == 2
        assert not mem._is_calibrated

    def test_mem_mitigation_preserves_distribution(self):
        mem = MeasurementErrorMitigation(n_qubits=2, method="least_squares")

        # Set up fake calibration
        mem.calibration_matrix = np.eye(4)
        mem.inverse_matrix = np.eye(4)
        mem._is_calibrated = True

        measured_probs = np.array([0.3, 0.2, 0.25, 0.25])
        mitigated = mem.mitigate(measured_probs)

        # Should be valid probability distribution
        assert np.isclose(np.sum(mitigated), 1.0)
        assert np.all(mitigated >= 0)


class TestQuantumAggregator:
    @pytest.mark.asyncio
    async def test_aggregator_initialization(self):
        from qedge.fl.aggregator import (
            QuantumGlobalAggregator,
            QuantumAggregatorConfig,
        )

        config = QuantumAggregatorConfig(n_qubits=4, vqc_layers=2)
        aggregator = QuantumGlobalAggregator(config)

        assert aggregator.vqc is not None
        assert aggregator.qka is not None
        assert aggregator.global_state is None

    @pytest.mark.asyncio
    async def test_classical_aggregation(self):
        from qedge.fl.aggregator import (
            QuantumGlobalAggregator,
            QuantumAggregatorConfig,
            LocalModelUpdate,
        )

        config = QuantumAggregatorConfig(n_qubits=4, vqc_layers=2)
        aggregator = QuantumGlobalAggregator(config)

        updates = [
            LocalModelUpdate(
                client_id="client_1",
                weights=np.ones(10),
                n_samples=100,
            ),
            LocalModelUpdate(
                client_id="client_2",
                weights=np.ones(10) * 2,
                n_samples=100,
            ),
        ]

        result = await aggregator.aggregate(updates, use_quantum=False)

        # Should be average of 1 and 2 = 1.5
        expected = np.ones(10) * 1.5
        assert_array_almost_equal(result.weights.flatten()[:10], expected)

    @pytest.mark.asyncio
    async def test_quantum_aggregation(self):
        from qedge.fl.aggregator import (
            QuantumGlobalAggregator,
            QuantumAggregatorConfig,
            LocalModelUpdate,
        )

        config = QuantumAggregatorConfig(
            n_qubits=4,
            vqc_layers=2,
            use_error_mitigation=False,
        )
        aggregator = QuantumGlobalAggregator(config)

        updates = [
            LocalModelUpdate(
                client_id=f"client_{i}",
                weights=np.random.randn(10),
                n_samples=100,
            )
            for i in range(3)
        ]

        result = await aggregator.aggregate(updates, use_quantum=True)

        assert result.round_number == 1
        assert result.quantum_embedding is not None
        assert "quantum_norm" in result.aggregation_metrics


# Run tests with: pytest tests/unit/test_quantum.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
