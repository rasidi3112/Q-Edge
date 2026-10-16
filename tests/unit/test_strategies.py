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


def _w(*values):
    return {"w": np.asarray(values, np.float32)}


def test_fedyogi_second_moment_moves_by_sign_not_decay():
    from qedge.fl.strategies import fedyogi

    zero = _w(0.0, 0.0)
    first, state = fedyogi(zero, [(_w(1.0, -2.0), 1)], None, lr=0.1)

    delta = np.array([1.0, -2.0])
    v0 = 1e-6
    v1 = v0 - 0.01 * delta**2 * np.sign(v0 - delta**2)
    m1 = 0.1 * delta
    np.testing.assert_allclose(first["w"], 0.1 * m1 / (np.sqrt(v1) + 1e-3), rtol=1e-5)
    np.testing.assert_allclose(state.v["w"], v1, rtol=1e-6)


def test_adaptive_strategies_follow_the_average_delta_direction():
    from qedge.fl.strategies import fedadam, fedyogi

    start = _w(0.0, 0.0)
    updates = [(_w(1.0, -1.0), 1), (_w(3.0, -3.0), 3)]
    for rule in (fedadam, fedyogi):
        weights, _ = rule(start, updates, None)
        assert weights["w"][0] > 0 > weights["w"][1]


def test_server_strategy_rejects_unknown_names_and_fedavg_options():
    from qedge.fl.strategies import ServerStrategy

    with pytest.raises(ValueError, match="Unknown"):
        ServerStrategy("scaffold")
    with pytest.raises(ValueError, match="no options"):
        ServerStrategy("fedavg", lr=0.1)
