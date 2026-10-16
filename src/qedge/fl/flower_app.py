"""Flower ClientApp and ServerApp built on the simulator's code.

The client handler trains with `qedge.fl.training.local_train` on its
Dirichlet partition and the server aggregates with
`qedge.fl.strategies.ServerStrategy`, so a Flower run and `run_simulation`
use the same rules. Uses the Message API of Flower 1.40 (``ClientApp``,
``ServerApp``, ``flwr run``); ``flwr.client.start_client`` is not used.

Run config keys (``pyproject.toml`` ``[tool.flwr.app.config]`` or
``--run-config``): ``dataset``, ``model``, ``clients``, ``alpha``, ``seed``,
``rounds``, ``strategy``, ``lr``, ``local-epochs``, ``batch-size``, ``mu``,
``server-lr``. Node config keys: ``partition-id``, ``num-partitions``, which
must equal ``clients``; the server waits until ``clients`` nodes connect.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from functools import cache
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    from flwr.app import (
        Array,
        ArrayRecord,
        ConfigRecord,
        Context,
        Message,
        MetricRecord,
        RecordDict,
    )
    from flwr.clientapp import ClientApp
    from flwr.serverapp import Grid, ServerApp
    from flwr.serverapp.strategy import FedAvg
except ImportError as exc:  # pragma: no cover - exercised in a subprocess test
    raise ImportError("Flower is required for qedge.fl.flower_app; install q-edge[fl]") from exc

from qedge.fl.models import build_model, get_weights, set_weights
from qedge.fl.partition import Dataset, dirichlet_partition, load_dataset
from qedge.fl.simulation import SimulationConfig
from qedge.fl.strategies import ServerStrategy
from qedge.fl.training import evaluate, local_train

logger = logging.getLogger(__name__)

_KEYS = {
    "dataset": "dataset",
    "model": "model",
    "clients": "clients",
    "alpha": "alpha",
    "seed": "seed",
    "rounds": "rounds",
    "strategy": "strategy",
    "lr": "lr",
    "local-epochs": "local_epochs",
    "batch-size": "batch_size",
    "mu": "mu",
    "server-lr": "server_lr",
}


def config_from_flower(run_config: Mapping[str, Any]) -> SimulationConfig:
    """Translate Flower run config keys into a `SimulationConfig`.

    Raises:
        ValueError: If a key is unknown or a value is invalid.
    """
    unknown = set(run_config) - set(_KEYS)
    if unknown:
        raise ValueError(f"Unknown run config keys {sorted(unknown)}")
    return SimulationConfig(**{_KEYS[key]: value for key, value in run_config.items()})


def arrays_to_record(weights: Mapping[str, NDArray[Any]]) -> ArrayRecord:
    """Wrap named NumPy arrays in an `ArrayRecord`."""
    return ArrayRecord(
        array_dict={name: Array(np.asarray(value)) for name, value in weights.items()}
    )


def record_to_arrays(record: ArrayRecord) -> dict[str, NDArray[Any]]:
    """Unwrap an `ArrayRecord` into named NumPy arrays."""
    return {name: np.asarray(array.numpy()) for name, array in record.items()}


@cache
def _data(dataset: str, seed: int) -> Dataset:
    return load_dataset(dataset, seed=seed)


def client_train(
    content: RecordDict, config: SimulationConfig, partition_id: int, server_round: int
) -> RecordDict:
    """Train one client on its partition starting from ``content["arrays"]``.

    Returns:
        ``arrays`` with the local weights and ``metrics`` with
        ``num-examples`` and ``train_loss``.
    """
    data = _data(config.dataset, config.seed)
    shard = dirichlet_partition(data.y_train, config.clients, config.alpha, seed=config.seed)[
        partition_id
    ]
    model = build_model(config.model, data.n_features, data.n_classes, seed=config.seed)
    set_weights(model, record_to_arrays(content["arrays"]))  # type: ignore[arg-type]
    seed = config.seed * 1_000_003 + server_round * 1_009 + partition_id
    loss = local_train(
        model, data.x_train[shard], data.y_train[shard], config.train_config(), seed=seed
    )
    return RecordDict(
        {
            "arrays": arrays_to_record(get_weights(model)),
            "metrics": MetricRecord({"num-examples": len(shard), "train_loss": loss}),
        }
    )


class QEdgeStrategy(FedAvg):
    """Flower strategy that delegates aggregation to a `ServerStrategy`.

    Sampling, messaging and logging come from Flower's ``FedAvg``; only
    ``aggregate_train`` changes.

    Args:
        server: The aggregation rule.
        **kwargs: Passed to Flower's ``FedAvg``.
    """

    def __init__(self, server: ServerStrategy, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.server = server
        self._current: dict[str, NDArray[Any]] | None = None

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Remember the global weights sent out this round, then defer to FedAvg."""
        self._current = record_to_arrays(arrays)
        config["server-round"] = server_round
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_contents(self, contents: list[RecordDict]) -> ArrayRecord:
        """Aggregate reply contents with the server strategy.

        Raises:
            RuntimeError: If called before `configure_train`.
        """
        if self._current is None:
            raise RuntimeError("configure_train has not run")
        updates = [
            (record_to_arrays(c["arrays"]), int(c["metrics"]["num-examples"]))  # type: ignore[arg-type]
            for c in contents
        ]
        self._current = self.server.step(self._current, updates)
        return arrays_to_record(self._current)

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate the replies that carry no error."""
        contents = [message.content for message in replies if not message.has_error()]
        if not contents:
            return None, None
        losses = [float(c["metrics"]["train_loss"]) for c in contents]  # type: ignore[arg-type]
        return self.aggregate_contents(contents), MetricRecord(
            {"train_loss": float(np.mean(losses))}
        )


client_app = ClientApp()


@client_app.train()
def train(message: Message, context: Context) -> Message:
    """ClientApp train handler.

    Raises:
        ValueError: If ``num-partitions`` differs from the run's ``clients``.
    """
    config = config_from_flower(context.run_config)
    partitions = int(context.node_config["num-partitions"])
    if partitions != config.clients:
        raise ValueError(f"num-partitions {partitions} != clients {config.clients}")
    server_round = int(message.content["config"]["server-round"])  # type: ignore[arg-type]
    content = client_train(
        message.content, config, int(context.node_config["partition-id"]), server_round
    )
    return Message(content=content, reply_to=message)


server_app = ServerApp()


@server_app.main()
def main(grid: Grid, context: Context) -> None:
    """ServerApp main: run ``rounds`` rounds with all ``clients`` nodes each round."""
    config = config_from_flower(context.run_config)
    data = _data(config.dataset, config.seed)
    model = build_model(config.model, data.n_features, data.n_classes, seed=config.seed)
    strategy = QEdgeStrategy(
        config.server_strategy(),
        fraction_train=1.0,
        fraction_evaluate=0.0,
        min_train_nodes=config.clients,
        min_available_nodes=config.clients,
    )

    def evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        set_weights(model, record_to_arrays(arrays))
        accuracy, loss = evaluate(model, data.x_test, data.y_test)
        logger.info("round %d: accuracy %.4f loss %.4f", server_round, accuracy, loss)
        return MetricRecord({"test_accuracy": accuracy, "test_loss": loss})

    strategy.start(
        grid=grid,
        initial_arrays=arrays_to_record(get_weights(model)),
        num_rounds=config.rounds,
        train_config=ConfigRecord({}),
        evaluate_fn=evaluate_fn,
    )
