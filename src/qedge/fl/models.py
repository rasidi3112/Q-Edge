"""Classical models for the FL simulator and weight conversion helpers.

Requires PyTorch (``q-edge[fl]``); importing without it raises `ImportError`
with that hint instead of failing later.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised in a subprocess test
    raise ImportError("PyTorch is required for qedge.fl.models; install q-edge[fl]") from exc

MODELS = ("logreg", "mlp", "classical-head", "hybrid-vqc")
HEAD_MODELS = ("classical-head", "hybrid-vqc")


def build_model(
    name: str, n_features: int, n_classes: int, *, seed: int, **options: Any
) -> nn.Module:
    """Build a classifier that outputs logits of shape ``(batch, n_classes)``.

    Args:
        name: ``"logreg"`` (one linear layer), ``"mlp"`` (one hidden ReLU
            layer), ``"hybrid-vqc"`` or ``"classical-head"``
            (`qedge.quantum.models`; these import PennyLane).
        n_features: Input width.
        n_classes: Number of classes.
        seed: Seed for weight initialization.
        **options: ``hidden`` for the MLP; `qedge.quantum.models.head_config`
            options for the head models.

    Raises:
        ValueError: If ``name`` is not one of `MODELS` or an option does not apply.
    """
    if name in HEAD_MODELS:
        from qedge.quantum.models import build_head_model, head_config

        try:
            config = head_config(**options)
        except TypeError as exc:
            raise ValueError(f"Invalid options for {name}: {exc}") from exc
        return build_head_model(name, n_features, n_classes, config, seed=seed)
    allowed = {"hidden"} if name == "mlp" else set()
    if set(options) - allowed:
        raise ValueError(f"Options {sorted(set(options) - allowed)} do not apply to {name!r}")
    hidden = int(options.get("hidden", 16))
    generator_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        if name == "logreg":
            model: nn.Module = nn.Linear(n_features, n_classes)
        elif name == "mlp":
            model = nn.Sequential(
                nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, n_classes)
            )
        else:
            raise ValueError(f"Unknown model {name!r}; expected {MODELS}")
    finally:
        torch.random.set_rng_state(generator_state)
    return model


def count_parameters(model: nn.Module) -> int:
    """Number of trainable scalars in ``model``."""
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)


def get_weights(model: nn.Module) -> dict[str, NDArray[Any]]:
    """Copy the model's state dict into float32 NumPy arrays, keyed by name."""
    return {
        name: tensor.detach().cpu().numpy().astype(np.float32, copy=True)
        for name, tensor in model.state_dict().items()
    }


def set_weights(model: nn.Module, weights: Mapping[str, NDArray[Any]]) -> None:
    """Load arrays produced by `get_weights` (or an aggregate of them).

    Raises:
        ValueError: If names or shapes differ from the model's state dict.
    """
    state = model.state_dict()
    if set(state) != set(weights):
        raise ValueError(f"Weight names {sorted(weights)} do not match {sorted(state)}")
    loaded = {}
    for name, tensor in state.items():
        array = np.asarray(weights[name])
        if tuple(array.shape) != tuple(tensor.shape):
            raise ValueError(f"{name}: shape {array.shape} != {tuple(tensor.shape)}")
        loaded[name] = torch.as_tensor(array, dtype=tensor.dtype)
    model.load_state_dict(loaded)
