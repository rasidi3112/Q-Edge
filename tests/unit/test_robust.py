import numpy as np
import pytest

from qedge.fl.robust import (
    DEFENSES,
    RobustAggregator,
    bounded_fedavg,
    coordinate_median,
    krum_scores,
    multi_krum,
    norm_clipped_fedavg,
    trimmed_mean,
)


def _u(*values, n=1):
    return ({"w": np.asarray(values, np.float32)}, n)


HONEST = [_u(1.0, 1.0), _u(1.1, 0.9), _u(0.9, 1.1), _u(1.0, 1.05), _u(1.05, 1.0)]
ATTACK = [_u(100.0, -100.0), _u(90.0, -80.0)]


def test_median_ignores_a_minority_of_outliers():
    result = coordinate_median(HONEST + ATTACK)

    np.testing.assert_allclose(result["w"], [1.05, 1.0])
    assert result["w"].dtype == np.float32


def test_trimmed_mean_drops_the_extremes_per_coordinate():
    updates = [_u(v) for v in (0.0, 1.0, 2.0, 3.0, 100.0)]

    np.testing.assert_allclose(trimmed_mean(updates, beta=0.2)["w"], [2.0])
    np.testing.assert_allclose(trimmed_mean(updates, beta=0.0)["w"], [21.2])
    with pytest.raises(ValueError, match="beta"):
        trimmed_mean(updates, beta=0.5)


def test_krum_selects_an_honest_update_and_multi_krum_averages_honest_ones():
    updates = HONEST + ATTACK

    chosen = multi_krum(updates, f=2, m=1)["w"]
    averaged = multi_krum(updates, f=2)["w"]

    assert any(np.allclose(chosen, w["w"]) for w, _ in HONEST)
    np.testing.assert_allclose(averaged, np.mean([w["w"] for w, _ in HONEST], axis=0), rtol=1e-5)


def test_krum_scores_sum_nearest_squared_distances():
    flat = np.array([[0.0], [1.0], [3.0], [10.0], [11.0]])

    # n - f - 2 = 2 neighbours for f = 1.
    np.testing.assert_allclose(krum_scores(flat, 1), [1 + 9, 1 + 4, 4 + 9, 1 + 49, 1 + 64])


def test_krum_needs_more_than_2f_plus_2_clients():
    with pytest.raises(ValueError, match="2f"):
        multi_krum(HONEST, f=2)


def test_norm_clip_bounds_each_delta_by_the_median_norm():
    origin = {"w": np.zeros(2, np.float32)}
    updates = [_u(1.0, 0.0), _u(0.0, 2.0), _u(300.0, 400.0)]

    result = norm_clipped_fedavg(origin, updates)

    # Median norm 2: the outlier is scaled to (1.2, 1.6).
    np.testing.assert_allclose(
        result["w"], [(1.0 + 0.0 + 1.2) / 3, (0.0 + 2.0 + 1.6) / 3], rtol=1e-5
    )
    np.testing.assert_allclose(
        norm_clipped_fedavg(origin, updates, clip=1.0)["w"],
        [(1 + 0 + 0.6) / 3, (0 + 1 + 0.8) / 3],
        rtol=1e-5,
    )


def test_bounded_sample_weights_cap_inflated_claims():
    updates = [_u(0.0, n=10), _u(0.0, n=12), _u(0.0, n=8), _u(9.0, n=1_000_000)]

    result = bounded_fedavg(updates, cap=3.0)

    # Cap 3 * median(10, 12, 8, 1e6) = 33.
    np.testing.assert_allclose(result["w"], [9.0 * 33 / (10 + 12 + 8 + 33)], rtol=1e-5)
    with pytest.raises(ValueError, match="cap"):
        bounded_fedavg(updates, cap=0.5)


def test_every_defense_returns_honest_like_weights_under_attack():
    origin = {"w": np.ones(2, np.float32)}
    honest_mean = np.mean([w["w"] for w, _ in HONEST], axis=0)
    for name in DEFENSES:
        if name == "bounded":
            continue  # bounds sample counts only; equal counts leave FedAvg unchanged
        options = {"krum": {"f": 2}, "multi_krum": {"f": 2}, "trimmed_mean": {"beta": 0.3}}
        result = RobustAggregator(name, **options.get(name, {})).step(origin, HONEST + ATTACK)
        assert np.abs(result["w"] - honest_mean).max() < 1.0, name


def test_robust_aggregator_rejects_unknown_names_and_options():
    with pytest.raises(ValueError, match="Unknown defense"):
        RobustAggregator("geomed")
    with pytest.raises(ValueError, match="do not apply"):
        RobustAggregator("median", beta=0.1)


def test_robust_rules_reject_mismatched_updates():
    with pytest.raises(ValueError, match="names"):
        coordinate_median([_u(1.0), ({"v": np.zeros(1)}, 1)])
    with pytest.raises(ValueError, match="at least one"):
        coordinate_median([])
