"""Local training and evaluation shared by the simulator and the Flower client.

FedProx (Li et al., "Federated Optimization in Heterogeneous Networks", 2018,
arXiv:1812.06127) is a client-side change: with ``mu > 0`` the local loss gets
``mu / 2 * ||w - w_global||^2``, which keeps local models near the global one
on non-IID data. The server still averages with FedAvg.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised in a subprocess test
    raise ImportError("PyTorch is required for qedge.fl.training; install q-edge[fl]") from exc


@dataclass(frozen=True)
class TrainConfig:
    """Client-side hyperparameters.

    Attributes:
        epochs: Passes over the local data per round.
        batch_size: Minibatch size.
        lr: SGD learning rate.
        mu: FedProx proximal coefficient; 0 disables the term.
    """

    epochs: int = 1
    batch_size: int = 16
    lr: float = 0.1
    mu: float = 0.0


def proximal_term(model: nn.Module, global_params: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """``0.5 * sum ||p - p_global||^2`` over the model's parameters."""
    total = torch.zeros(())
    for name, param in model.named_parameters():
        total = total + torch.sum((param - global_params[name]) ** 2)
    return 0.5 * total


def local_train(
    model: nn.Module,
    x: NDArray[np.float32],
    y: NDArray[np.int64],
    config: TrainConfig,
    *,
    seed: int,
) -> float:
    """Train ``model`` in place with SGD on one client's data.

    Returns:
        Mean training loss over the last epoch (without the proximal term).
    """
    generator = torch.Generator().manual_seed(seed)
    features, labels = torch.from_numpy(x), torch.from_numpy(y)
    global_params = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    epoch_loss = 0.0
    for _ in range(config.epochs):
        order = torch.randperm(len(labels), generator=generator)
        epoch_loss, seen = 0.0, 0
        for start in range(0, len(labels), config.batch_size):
            batch = order[start : start + config.batch_size]
            optimizer.zero_grad()
            loss = criterion(model(features[batch]), labels[batch])
            objective = (
                loss + config.mu * proximal_term(model, global_params) if config.mu else loss
            )
            objective.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(batch)
            seen += len(batch)
        epoch_loss /= max(seen, 1)
    return epoch_loss


def evaluate(model: nn.Module, x: NDArray[np.float32], y: NDArray[np.int64]) -> tuple[float, float]:
    """Return ``(accuracy, mean cross-entropy loss)`` of ``model`` on ``(x, y)``."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x))
        labels = torch.from_numpy(y)
        loss = float(nn.functional.cross_entropy(logits, labels))
        accuracy = float((logits.argmax(dim=1) == labels).float().mean())
    return accuracy, loss


def weight_delta(
    new: Mapping[str, NDArray[Any]], old: Mapping[str, NDArray[Any]]
) -> dict[str, NDArray[Any]]:
    """``new - old`` per tensor."""
    return {name: np.asarray(new[name]) - np.asarray(old[name]) for name in old}
