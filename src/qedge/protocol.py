"""Message formats shared by the server and its clients."""

from __future__ import annotations

from typing import Any

from qedge.fl.weights import ModelSpec, TensorSpec

CHALLENGE_LABEL = b"qedge/auth/v1"
SCOPES = ("client", "admin")


def challenge_message(nonce: bytes, client_id: str, server_id: str) -> bytes:
    """Bytes a client signs to answer a login challenge.

    ``label || len(nonce) nonce || len(client_id) client_id || len(server_id)
    server_id`` with 2-byte big-endian lengths.
    """
    parts = [nonce, client_id.encode(), server_id.encode()]
    return CHALLENGE_LABEL + b"".join(len(part).to_bytes(2, "big") + part for part in parts)


def spec_to_json(spec: ModelSpec) -> list[dict[str, Any]]:
    """Encode a spec as a list of ``{name, shape, dtype}`` objects."""
    return [{"name": t.name, "shape": list(t.shape), "dtype": t.dtype} for t in spec.tensors]


def spec_from_json(data: list[dict[str, Any]]) -> ModelSpec:
    """Inverse of `spec_to_json`.

    Raises:
        ValueError: If an entry is malformed.
        KeyError: If an entry lacks a field.
    """
    return ModelSpec(
        tuple(
            TensorSpec(str(t["name"]), tuple(int(d) for d in t["shape"]), t["dtype"]) for t in data
        )
    )
