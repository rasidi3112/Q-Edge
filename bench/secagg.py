"""Cost of one secure-aggregation round for 5, 10, 20 and 50 clients.

For each client count and model size (1 210 parameters, the digits MLP, and
100 000), runs `qedge.fl.secagg.SecureAggregator` ``--repeats`` times on
random updates and reports the median per-client bytes and seconds, the
server's seconds, and the bytes a plain float32 upload would take. Pair
seeds use X25519 + ML-KEM-768 and signatures ML-DSA-65 from ``--backend``.
Times are single-process wall clock without network delay.

Writes results/summary/secagg.json (or ``--output``).

Usage:
    python bench/secagg.py [--backend liboqs] [--clients 5 10 20 50] [--output PATH]
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from qedge.crypto.backends import get_kem, get_signer
from qedge.fl.secagg import SecureAggregator

SIZES = (1210, 100_000)
PACKAGES = ("q-edge", "numpy", "cryptography", "liboqs-python", "kyber-py", "dilithium-py")


def _environment() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "packages": versions,
    }


def measure(backend: str, n_clients: int, size: int, repeats: int) -> dict[str, Any]:
    """Median cost over ``repeats`` rounds."""
    rng = np.random.default_rng(0)
    ids = [f"c{i:03d}" for i in range(n_clients)]
    agg = SecureAggregator(
        ids, kem=get_kem("X25519-ML-KEM-768", backend), signer=get_signer("ML-DSA-65", backend)
    )
    costs = []
    for round_id in range(repeats):
        updates = {
            cid: ({"w": rng.normal(size=size).astype(np.float32)}, int(rng.integers(10, 100)))
            for cid in ids
        }
        agg.aggregate(round_id, updates)
        costs.append(agg.last_cost)
    return {
        "clients": n_clients,
        "parameters": size,
        "bytes_up_per_client": statistics.median(c.bytes_up_per_client for c in costs),
        "bytes_down_per_client": statistics.median(c.bytes_down_per_client for c in costs),
        "plain_bytes_up_per_client": 4 * size,
        "client_seconds": statistics.median(c.client_seconds for c in costs),
        "server_seconds": statistics.median(c.server_seconds for c in costs),
    }


def main() -> None:
    """Run every configuration and write the summary JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="liboqs")
    parser.add_argument("--clients", type=int, nargs="+", default=[5, 10, 20, 50])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results/summary/secagg.json"))
    args = parser.parse_args()

    rows = []
    for n in args.clients:
        for size in SIZES:
            row = measure(args.backend, n, size, args.repeats)
            rows.append(row)
            print(
                f"{n:3d} clients {size:7d} params: up {row['bytes_up_per_client'] / 1024:8.1f} KiB "
                f"(plain {row['plain_bytes_up_per_client'] / 1024:7.1f}), "
                f"client {row['client_seconds'] * 1000:7.1f} ms, "
                f"server {row['server_seconds'] * 1000:8.1f} ms"
            )
    summary = {
        "description": __doc__.splitlines()[0],
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "backend": args.backend,
        "kem": "X25519-ML-KEM-768",
        "signature": "ML-DSA-65",
        "ring_bits": 64,
        "repeats": args.repeats,
        "environment": _environment(),
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
