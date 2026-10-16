import numpy as np
import pytest

from qedge.quantum.diagnostics import (
    ansatz_expressibility,
    gradient_variance,
    haar_fidelity_histogram,
    kernel_concentration,
    scaling,
)
from qedge.quantum.kernels import QuantumKernel


def test_haar_histogram_sums_to_one():
    assert haar_fidelity_histogram(3, 20).sum() == pytest.approx(1.0)


def test_gradient_variance_decreases_with_width():
    variances = [gradient_variance(n, 6, samples=60, seed=1) for n in (2, 4, 6)]

    assert variances[0] > variances[1] > variances[2] > 0
    assert scaling(variances) < -0.5


def test_more_layers_are_more_expressive():
    shallow = ansatz_expressibility(3, 1, pairs=200)
    deeper = ansatz_expressibility(3, 3, pairs=200)

    assert deeper < shallow


def test_kernel_concentration_reports_off_diagonal_statistics():
    x = np.random.default_rng(0).normal(size=(6, 2))
    stats = kernel_concentration(QuantumKernel(2), x)

    assert 0 < stats["mean"] < 1 and stats["variance"] > 0


def test_gradient_variance_rejects_unknown_observables():
    with pytest.raises(ValueError, match="observable"):
        gradient_variance(2, 1, observable="mixed")
