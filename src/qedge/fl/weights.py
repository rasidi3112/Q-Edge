"""Explicit wire format for model weights, with no object serialization.

Both sides agree on a `ModelSpec` (tensor names, shapes and dtypes) before any
weights move, so the payload carries no structure of its own:

    payload = b"QEW1" || sha256(spec) (32 bytes) || tensor bytes in spec order

Tensors are little-endian float32 or float64, C order. `decode_weights` checks
the magic, the spec digest and the exact length before reading, and rejects
NaN and infinity, so a malformed payload can only raise `WeightFormatError`.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

MAGIC = b"QEW1"
_DTYPES: dict[str, np.dtype[Any]] = {"float32": np.dtype("<f4"), "float64": np.dtype("<f8")}


class WeightFormatError(ValueError):
    """Weights do not match the agreed `ModelSpec` or the payload is malformed."""


@dataclass(frozen=True)
class TensorSpec:
    """Name, shape and dtype of one tensor.

    Raises:
        ValueError: If the name is empty, a dimension is negative, or the dtype
            is not float32 or float64.
    """

    name: str
    shape: tuple[int, ...]
    dtype: Literal["float32", "float64"]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Tensor name must not be empty")
        if any(dim < 0 for dim in self.shape):
            raise ValueError(f"{self.name}: negative dimension in {self.shape}")
        if self.dtype not in _DTYPES:
            raise ValueError(f"{self.name}: dtype must be float32 or float64, got {self.dtype}")

    @property
    def nbytes(self) -> int:
        """Encoded size in bytes."""
        return int(np.prod(self.shape, dtype=np.int64)) * _DTYPES[self.dtype].itemsize


@dataclass(frozen=True)
class ModelSpec:
    """Ordered tensor layout shared by client and server.

    Raises:
        ValueError: If tensor names are not unique.
    """

    tensors: tuple[TensorSpec, ...]

    def __post_init__(self) -> None:
        names = [tensor.name for tensor in self.tensors]
        if len(set(names)) != len(names):
            raise ValueError(f"Tensor names must be unique: {names}")

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, NDArray[Any]]) -> ModelSpec:
        """Describe ``arrays`` in iteration order.

        Raises:
            ValueError: If an array is not float32 or float64.
        """
        return cls(
            tuple(
                TensorSpec(name, tuple(int(d) for d in array.shape), _dtype_name(array))
                for name, array in arrays.items()
            )
        )

    @property
    def nbytes(self) -> int:
        """Size of an encoded payload in bytes, header included."""
        return len(MAGIC) + 32 + sum(tensor.nbytes for tensor in self.tensors)

    def digest(self) -> bytes:
        """SHA-256 over a canonical JSON encoding of the spec."""
        canonical = json.dumps(
            [[t.name, list(t.shape), t.dtype] for t in self.tensors], separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode()).digest()


def _dtype_name(array: NDArray[Any]) -> Literal["float32", "float64"]:
    if array.dtype == np.float32:
        return "float32"
    if array.dtype == np.float64:
        return "float64"
    raise ValueError(f"Unsupported dtype {array.dtype}; expected float32 or float64")


def encode_weights(spec: ModelSpec, arrays: Mapping[str, NDArray[Any]]) -> bytes:
    """Encode ``arrays`` in the layout of ``spec``.

    Raises:
        WeightFormatError: If names, shapes or dtypes differ from ``spec`` or a
            value is not finite.
    """
    expected = [tensor.name for tensor in spec.tensors]
    if sorted(arrays) != sorted(expected):
        raise WeightFormatError(f"Tensor names {sorted(arrays)} do not match spec {expected}")
    parts = [MAGIC, spec.digest()]
    for tensor in spec.tensors:
        array = np.asarray(arrays[tensor.name])
        if tuple(array.shape) != tensor.shape:
            raise WeightFormatError(f"{tensor.name}: shape {array.shape} != spec {tensor.shape}")
        if array.dtype != _DTYPES[tensor.dtype]:
            raise WeightFormatError(f"{tensor.name}: dtype {array.dtype} != spec {tensor.dtype}")
        if not np.all(np.isfinite(array)):
            raise WeightFormatError(f"{tensor.name}: contains NaN or infinity")
        parts.append(np.ascontiguousarray(array, dtype=_DTYPES[tensor.dtype]).tobytes())
    return b"".join(parts)


def decode_weights(spec: ModelSpec, payload: bytes) -> dict[str, NDArray[Any]]:
    """Decode a payload produced by `encode_weights` for the same ``spec``.

    Returns:
        Writable arrays keyed by tensor name, in spec order.

    Raises:
        WeightFormatError: If the payload is malformed, was encoded for another
            spec, has the wrong length, or contains NaN or infinity.
    """
    if len(payload) != spec.nbytes:
        raise WeightFormatError(f"Payload is {len(payload)} bytes; spec needs {spec.nbytes}")
    if payload[: len(MAGIC)] != MAGIC:
        raise WeightFormatError("Payload does not start with the QEW1 magic")
    offset = len(MAGIC)
    if payload[offset : offset + 32] != spec.digest():
        raise WeightFormatError("Payload was encoded for a different model spec")
    offset += 32
    arrays: dict[str, NDArray[Any]] = {}
    for tensor in spec.tensors:
        raw = payload[offset : offset + tensor.nbytes]
        offset += tensor.nbytes
        array = np.frombuffer(raw, dtype=_DTYPES[tensor.dtype]).reshape(tensor.shape).copy()
        if not np.all(np.isfinite(array)):
            raise WeightFormatError(f"{tensor.name}: contains NaN or infinity")
        arrays[tensor.name] = array
    return arrays


UPDATE_MAGIC = b"QEU1"
_UPDATE_HEADER = struct.Struct(">4sQId")


@dataclass(frozen=True)
class UpdateHeader:
    """Round id and the client's claimed sample count and local loss."""

    round_id: int
    n_samples: int
    local_loss: float


def encode_update(
    header: UpdateHeader, spec: ModelSpec, arrays: Mapping[str, NDArray[Any]]
) -> bytes:
    """Encode a client update: ``QEU1 | round u64 | n_samples u32 | loss f64 | weights``.

    Raises:
        WeightFormatError: As `encode_weights`, or if the header is out of range.
    """
    if not 0 <= header.round_id < 2**64 or not 0 <= header.n_samples < 2**32:
        raise WeightFormatError("Round id or sample count out of range")
    if not math.isfinite(header.local_loss):
        raise WeightFormatError("Local loss is not finite")
    packed = _UPDATE_HEADER.pack(UPDATE_MAGIC, header.round_id, header.n_samples, header.local_loss)
    return packed + encode_weights(spec, arrays)


def decode_update(spec: ModelSpec, payload: bytes) -> tuple[UpdateHeader, dict[str, NDArray[Any]]]:
    """Decode a payload produced by `encode_update` for ``spec``.

    Raises:
        WeightFormatError: If the header or the weights are malformed.
    """
    if len(payload) < _UPDATE_HEADER.size or payload[:4] != UPDATE_MAGIC:
        raise WeightFormatError("Payload does not start with a QEU1 update header")
    _, round_id, n_samples, local_loss = _UPDATE_HEADER.unpack_from(payload)
    if not math.isfinite(local_loss):
        raise WeightFormatError("Local loss is not finite")
    arrays = decode_weights(spec, payload[_UPDATE_HEADER.size :])
    return UpdateHeader(round_id, n_samples, local_loss), arrays
