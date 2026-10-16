"""Datasets and non-IID client partitions.

Datasets come from scikit-learn (bundled, no download): ``breast_cancer``
(569 samples, 30 features, 2 classes) and ``digits`` (1797 samples, 64
features, 10 classes). Features are standardized with statistics from the
training split only. Labels are split across clients with a Dirichlet
distribution per class, the scheme of Hsu et al. ("Measuring the Effects of
Non-Identical Data Distribution for Federated Visual Classification", 2019):
small ``alpha`` gives each client few classes, large ``alpha`` approaches IID.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

DATASETS = ("breast_cancer", "digits")


@dataclass(frozen=True)
class Dataset:
    """A train/test split with standardized float32 features and int64 labels."""

    name: str
    x_train: NDArray[np.float32]
    y_train: NDArray[np.int64]
    x_test: NDArray[np.float32]
    y_test: NDArray[np.int64]
    n_classes: int

    @property
    def n_features(self) -> int:
        """Number of input features."""
        return int(self.x_train.shape[1])


def load_dataset(name: str, *, seed: int, test_fraction: float = 0.2) -> Dataset:
    """Load a scikit-learn dataset and split it, stratified by label.

    Raises:
        ValueError: If ``name`` is not one of `DATASETS`.
        ImportError: If scikit-learn is not installed (``q-edge[fl]``).
    """
    try:
        from sklearn.datasets import load_breast_cancer, load_digits
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise ImportError("scikit-learn is required; install q-edge[fl]") from exc
    loaders: dict[str, Any] = {"breast_cancer": load_breast_cancer, "digits": load_digits}
    if name not in loaders:
        raise ValueError(f"Unknown dataset {name!r}; expected {DATASETS}")
    bunch = loaders[name]()
    x, y = np.asarray(bunch.data, dtype=np.float64), np.asarray(bunch.target, dtype=np.int64)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_fraction, random_state=seed, stratify=y
    )
    mean, std = x_train.mean(axis=0), x_train.std(axis=0)
    std[std == 0] = 1.0
    return Dataset(
        name=name,
        x_train=((x_train - mean) / std).astype(np.float32),
        y_train=y_train,
        x_test=((x_test - mean) / std).astype(np.float32),
        y_test=y_test,
        n_classes=int(y.max()) + 1,
    )


def dirichlet_partition(
    labels: NDArray[np.int64],
    n_clients: int,
    alpha: float,
    *,
    seed: int,
    min_size: int = 2,
    max_attempts: int = 1000,
) -> list[NDArray[np.int64]]:
    """Split sample indices across clients with per-class Dirichlet proportions.

    Args:
        labels: Label of every training sample.
        n_clients: Number of clients.
        alpha: Dirichlet concentration; must be positive.
        seed: Seed for the draws.
        min_size: Fewest samples any client may receive; draws are repeated
            until every client has at least this many.
        max_attempts: Draws before giving up.

    Returns:
        One sorted index array per client; together they cover every sample
        exactly once.

    Raises:
        ValueError: If the arguments are invalid or no draw satisfies
            ``min_size`` within ``max_attempts``.
    """
    if n_clients < 1 or alpha <= 0:
        raise ValueError("n_clients must be >= 1 and alpha > 0")
    if n_clients * min_size > len(labels):
        raise ValueError(f"{len(labels)} samples cannot give {n_clients} clients {min_size} each")
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    for _ in range(max_attempts):
        shards: list[list[int]] = [[] for _ in range(n_clients)]
        for cls in classes:
            indices = rng.permutation(np.flatnonzero(labels == cls))
            proportions = rng.dirichlet(np.full(n_clients, alpha))
            cuts = (np.cumsum(proportions)[:-1] * len(indices)).astype(int)
            for client, part in enumerate(np.split(indices, cuts)):
                shards[client].extend(part.tolist())
        if min(len(shard) for shard in shards) >= min_size:
            return [np.sort(np.asarray(shard, dtype=np.int64)) for shard in shards]
    raise ValueError(
        f"No Dirichlet(alpha={alpha}) draw gave every client {min_size} samples "
        f"in {max_attempts} attempts; lower min_size or use fewer clients"
    )
