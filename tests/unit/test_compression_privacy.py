import numpy as np
import pytest

from qedge.fl.compression import TopKCompressor, dense_bytes
from qedge.fl.privacy import clip_and_noise


def test_topk_keeps_the_largest_entries_and_counts_bytes():
    sent, nbytes = TopKCompressor(0.25).compress({"w": np.array([0.1, -5.0, 0.2, 3.0] * 2)})

    np.testing.assert_array_equal(np.flatnonzero(sent["w"]), [1, 5])
    assert nbytes == 2 * 8
    assert dense_bytes({"w": np.zeros(8)}) == 32


def test_error_feedback_eventually_sends_everything():
    rng = np.random.default_rng(0)
    compressor = TopKCompressor(0.1)
    deltas = [{"w": rng.normal(size=50)} for _ in range(20)]

    total_sent = sum(compressor.compress(d)[0]["w"] for d in deltas)
    total_true = sum(d["w"] for d in deltas)

    np.testing.assert_allclose(total_sent + compressor._residual["w"], total_true, atol=1e-9)


def test_topk_rejects_ratio_outside_unit_interval():
    for ratio in (0.0, 1.5):
        with pytest.raises(ValueError):
            TopKCompressor(ratio)


def test_clipping_bounds_the_norm_and_leaves_small_updates_alone():
    rng = np.random.default_rng(0)
    big = {"a": np.full(4, 10.0), "b": np.full(4, -10.0)}

    clipped, norm = clip_and_noise(big, clip=1.0, noise_multiplier=0.0, rng=rng)

    assert norm == pytest.approx(np.sqrt(800))
    total = np.sqrt(sum(np.sum(v**2) for v in clipped.values()))
    assert total == pytest.approx(1.0)
    small = {"a": np.full(4, 0.01)}
    np.testing.assert_allclose(
        clip_and_noise(small, clip=1.0, noise_multiplier=0.0, rng=rng)[0]["a"], 0.01
    )


def test_noise_has_standard_deviation_multiplier_times_clip():
    noisy, _ = clip_and_noise(
        {"w": np.zeros(20000)}, clip=0.5, noise_multiplier=2.0, rng=np.random.default_rng(1)
    )

    assert np.std(noisy["w"]) == pytest.approx(1.0, rel=0.03)
