import numpy as np
import pytest

from qedge.fl.strategies import fedavg


def test_fedavg_weights_by_sample_count():
    a = {"w": np.array([1.0, 1.0], np.float32), "b": np.array([0.0], np.float32)}
    b = {"w": np.array([3.0, 3.0], np.float32), "b": np.array([4.0], np.float32)}

    result = fedavg([(a, 1), (b, 3)])

    np.testing.assert_allclose(result["w"], [2.5, 2.5])
    np.testing.assert_allclose(result["b"], [3.0])
    assert result["w"].dtype == np.float32


def test_fedavg_rejects_empty_zero_weight_and_mismatched_updates():
    a = {"w": np.zeros(2)}
    with pytest.raises(ValueError, match="at least one"):
        fedavg([])
    with pytest.raises(ValueError, match="positive"):
        fedavg([(a, 0)])
    with pytest.raises(ValueError, match="names"):
        fedavg([(a, 1), ({"v": np.zeros(2)}, 1)])
