import numpy as np
import pytest

pytest.importorskip("sklearn")

from qedge.fl.partition import DATASETS, dirichlet_partition, load_dataset


@pytest.mark.parametrize("name", DATASETS)
def test_datasets_are_standardized_with_train_statistics(name):
    data = load_dataset(name, seed=0)

    np.testing.assert_allclose(data.x_train.mean(axis=0), 0.0, atol=1e-5)
    assert data.x_train.dtype == np.float32 and data.y_train.dtype == np.int64
    assert len(data.y_test) == pytest.approx(0.2 * (len(data.y_train) + len(data.y_test)), abs=1)
    assert set(np.unique(data.y_test)) == set(range(data.n_classes))


def test_unknown_dataset_is_rejected():
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_dataset("mnist", seed=0)


@pytest.mark.parametrize("alpha", [0.1, 0.5, 100.0])
def test_partition_covers_every_sample_once_and_is_deterministic(alpha):
    labels = np.repeat(np.arange(4), 50)

    shards = dirichlet_partition(labels, 8, alpha, seed=3)

    np.testing.assert_array_equal(np.sort(np.concatenate(shards)), np.arange(200))
    assert min(len(s) for s in shards) >= 2
    again = dirichlet_partition(labels, 8, alpha, seed=3)
    assert all(np.array_equal(a, b) for a, b in zip(shards, again, strict=True))


def test_small_alpha_gives_more_skewed_label_mixes_than_large_alpha():
    labels = np.repeat(np.arange(4), 250)

    def mean_max_share(alpha):
        shards = dirichlet_partition(labels, 10, alpha, seed=0)
        return np.mean([np.bincount(labels[s], minlength=4).max() / len(s) for s in shards])

    assert mean_max_share(0.1) > 0.7
    assert mean_max_share(100.0) < 0.35


def test_partition_rejects_impossible_requests():
    labels = np.zeros(10, np.int64)
    with pytest.raises(ValueError, match="cannot give"):
        dirichlet_partition(labels, 6, 0.5, seed=0)
    with pytest.raises(ValueError, match="alpha"):
        dirichlet_partition(labels, 2, 0.0, seed=0)
