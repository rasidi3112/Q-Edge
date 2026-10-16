"""In-process federated simulation on scikit-learn datasets.

Every round, each client trains the current global model on its Dirichlet
partition, sends ``w_local - w_global`` (optionally clipped and noised, then
top-k compressed), and the server applies a `ServerStrategy`. The global
model is evaluated on a held-out test split after every round. The same
training and strategy code backs the Flower apps in `qedge.fl.flower_app`.

Outputs go to ``<out_dir>/<run_id>/``: ``config.yaml`` (configuration,
partition sizes, package versions, machine) and ``metrics.jsonl`` (one JSON
object per round).
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import logging
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from qedge.fl.compression import TopKCompressor, dense_bytes
from qedge.fl.models import build_model, count_parameters, get_weights, set_weights
from qedge.fl.partition import Dataset, dirichlet_partition, load_dataset
from qedge.fl.privacy import clip_and_noise
from qedge.fl.strategies import Aggregator, ServerStrategy
from qedge.fl.training import TrainConfig, evaluate, local_train, weight_delta

logger = logging.getLogger(__name__)

STRATEGIES = ("fedavg", "fedprox", "fedadam", "fedyogi")
PACKAGES = ("q-edge", "numpy", "torch", "scikit-learn", "pennylane", "flwr")


@dataclass(frozen=True)
class SimulationConfig:
    """Everything that determines a run; stored in ``config.yaml``.

    ``strategy`` ``"fedprox"`` means FedAvg on the server with the proximal
    term ``mu`` in the client loss; ``mu`` must be 0 for other strategies.
    ``compression`` is the top-k ratio (None sends dense updates). ``dp_clip``
    enables clipping, ``dp_noise`` is the noise multiplier.
    """

    dataset: str = "breast_cancer"
    model: str = "logreg"
    clients: int = 10
    rounds: int = 30
    alpha: float = 0.5
    strategy: str = "fedavg"
    seed: int = 0
    local_epochs: int = 1
    batch_size: int = 16
    lr: float = 0.1
    mu: float = 0.0
    server_lr: float = 0.05
    compression: float | None = None
    dp_clip: float | None = None
    dp_noise: float = 0.0
    model_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy {self.strategy!r}; expected {STRATEGIES}")
        if self.strategy != "fedprox" and self.mu:
            raise ValueError("mu > 0 is FedProx; use strategy='fedprox'")
        if self.clients < 1 or self.rounds < 1:
            raise ValueError("clients and rounds must be at least 1")
        if self.dp_noise and self.dp_clip is None:
            raise ValueError("dp_noise needs dp_clip")

    def train_config(self) -> TrainConfig:
        """Client hyperparameters; FedProx defaults to mu = 0.01 when unset."""
        mu = self.mu if self.mu or self.strategy != "fedprox" else 0.01
        return TrainConfig(self.local_epochs, self.batch_size, self.lr, mu)

    def server_strategy(self) -> ServerStrategy:
        """The server rule for this run."""
        if self.strategy in ("fedavg", "fedprox"):
            return ServerStrategy("fedavg")
        return ServerStrategy(self.strategy, lr=self.server_lr)

    def run_id(self) -> str:
        """Directory name: UTC time, dataset, model, strategy, clients, alpha, seed."""
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        return (
            f"{stamp}-{self.dataset}-{self.model}-{self.strategy}"
            f"-c{self.clients}-a{self.alpha:g}-s{self.seed}"
        )


ModelFactory = Callable[[SimulationConfig, Dataset], Any]


def _default_model(config: SimulationConfig, data: Dataset) -> Any:
    return build_model(
        config.model, data.n_features, data.n_classes, seed=config.seed, **config.model_options
    )


def environment() -> dict[str, Any]:
    """Python, platform and package versions for run records."""
    versions: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
    }


def run_simulation(
    config: SimulationConfig,
    out_dir: Path | None = None,
    *,
    model_factory: ModelFactory = _default_model,
    aggregator: Aggregator | None = None,
) -> dict[str, Any]:
    """Run one simulation and optionally write its records.

    Args:
        config: The run configuration.
        out_dir: Parent directory for ``<run_id>/``; None writes nothing.
        model_factory: Builds the model from the config and dataset.
        aggregator: Server rule; defaults to ``config.server_strategy()``.

    Returns:
        Summary with ``final_accuracy``, ``best_accuracy``, ``bytes_up``,
        ``bytes_down``, ``parameters``, ``seconds``, ``run_dir`` and the
        per-round ``history``.
    """
    started = time.perf_counter()
    data = load_dataset(config.dataset, seed=config.seed)
    shards = dirichlet_partition(data.y_train, config.clients, config.alpha, seed=config.seed)
    model = model_factory(config, data)
    global_weights = get_weights(model)
    strategy = aggregator if aggregator is not None else config.server_strategy()
    train_config = config.train_config()
    compressors = (
        [TopKCompressor(config.compression) for _ in shards] if config.compression else None
    )
    noise_rng = np.random.default_rng(config.seed + 1)
    partition = [
        {
            "client": i,
            "samples": len(shard),
            "classes": np.bincount(data.y_train[shard], minlength=data.n_classes).tolist(),
        }
        for i, shard in enumerate(shards)
    ]

    run_dir = None
    if out_dir is not None:
        run_dir = out_dir / config.run_id()
        run_dir.mkdir(parents=True, exist_ok=False)
        record = {
            "config": dataclasses.asdict(config),
            "parameters": count_parameters(model),
            "model": model.describe() if hasattr(model, "describe") else None,
            "partition": partition,
            "test_samples": len(data.y_test),
            "environment": environment(),
        }
        (run_dir / "config.yaml").write_text(yaml.safe_dump(record, sort_keys=False))

    history: list[dict[str, Any]] = []
    total_up = total_down = 0
    for round_index in range(1, config.rounds + 1):
        round_start = time.perf_counter()
        updates: list[tuple[dict[str, NDArray[Any]], int]] = []
        losses, bytes_up = [], 0
        for client, shard in enumerate(shards):
            set_weights(model, global_weights)
            seed = config.seed * 1_000_003 + round_index * 1_009 + client
            losses.append(
                local_train(
                    model, data.x_train[shard], data.y_train[shard], train_config, seed=seed
                )
            )
            delta = weight_delta(get_weights(model), global_weights)
            if config.dp_clip is not None:
                delta, _ = clip_and_noise(
                    delta, clip=config.dp_clip, noise_multiplier=config.dp_noise, rng=noise_rng
                )
            if compressors is not None:
                delta, sent = compressors[client].compress(delta)
            else:
                sent = dense_bytes(delta)
            bytes_up += sent
            client_weights = {n: global_weights[n] + delta[n] for n in global_weights}
            updates.append((client_weights, len(shard)))
        global_weights = strategy.step(global_weights, updates)
        set_weights(model, global_weights)
        accuracy, loss = evaluate(model, data.x_test, data.y_test)
        bytes_down = dense_bytes(global_weights) * len(shards)
        total_up += bytes_up
        total_down += bytes_down
        row = {
            "round": round_index,
            "test_accuracy": accuracy,
            "test_loss": loss,
            "train_loss_mean": float(np.mean(losses)),
            "bytes_up": bytes_up,
            "bytes_down": bytes_down,
            "seconds": time.perf_counter() - round_start,
        }
        history.append(row)
        if run_dir is not None:
            with (run_dir / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
        logger.info("round %d: accuracy %.4f loss %.4f", round_index, accuracy, loss)

    return {
        "final_accuracy": history[-1]["test_accuracy"],
        "best_accuracy": max(row["test_accuracy"] for row in history),
        "bytes_up": total_up,
        "bytes_down": total_down,
        "parameters": count_parameters(model),
        "model": model.describe() if hasattr(model, "describe") else None,
        "seconds": time.perf_counter() - started,
        "run_dir": None if run_dir is None else str(run_dir),
        "history": history,
    }
