"""Time and size every available KEM and signature backend.

Writes results/summary/crypto.json (or ``--output``). Times are wall-clock
microseconds from ``time.perf_counter_ns`` after one warm-up call; each
operation runs ``--iterations`` times per backend (default 200 for liboqs and
20 for the much slower pure-Python backend).

Usage:
    python bench/crypto.py [--iterations N] [--output PATH] [--note TEXT]
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qedge.crypto.backends import available_backends, get_kem, get_signer
from qedge.crypto.transport import HEADER_SIZE, ClientCredentials, ServerIdentity, accept, initiate

KEMS = ("ML-KEM-768", "ML-KEM-1024", "X25519-ML-KEM-768")
SIGNATURES = ("ML-DSA-65", "ML-DSA-87")
DEFAULT_ITERATIONS = {"liboqs": 200, "pure": 20}
PACKAGES = ("q-edge", "cryptography", "kyber-py", "dilithium-py", "liboqs-python", "numpy")


def _timing(operation: Callable[[], object], iterations: int) -> dict[str, float | int]:
    operation()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        operation()
        samples.append((time.perf_counter_ns() - start) / 1e3)
    samples.sort()
    return {
        "iterations": iterations,
        "median_us": round(statistics.median(samples), 1),
        "p90_us": round(samples[int(0.9 * (len(samples) - 1))], 1),
        "min_us": round(samples[0], 1),
    }


def _bench_kem(name: str, backend: str, iterations: int) -> dict[str, Any]:
    kem = get_kem(name, backend)
    public_key, secret_key = kem.generate_keypair()
    ciphertext, _ = kem.encapsulate(public_key)
    return {
        "sizes_bytes": {
            "public_key": kem.public_key_size,
            "secret_key": kem.secret_key_size,
            "ciphertext": kem.ciphertext_size,
            "shared_secret": kem.shared_secret_size,
        },
        "keygen": _timing(kem.generate_keypair, iterations),
        "encapsulate": _timing(lambda: kem.encapsulate(public_key), iterations),
        "decapsulate": _timing(lambda: kem.decapsulate(secret_key, ciphertext), iterations),
    }


def _bench_signature(name: str, backend: str, iterations: int) -> dict[str, Any]:
    signer = get_signer(name, backend)
    public_key, secret_key = signer.generate_keypair()
    message = b"\0" * 1024
    signature = signer.sign(secret_key, message)
    return {
        "sizes_bytes": {
            "public_key": signer.public_key_size,
            "secret_key": signer.secret_key_size,
            "signature": signer.signature_size,
        },
        "message_bytes": len(message),
        "keygen": _timing(signer.generate_keypair, iterations),
        "sign": _timing(lambda: signer.sign(secret_key, message), iterations),
        "verify": _timing(lambda: signer.verify(public_key, message, signature), iterations),
    }


def _bench_transport(backend: str, iterations: int) -> dict[str, Any]:
    kem = get_kem("X25519-ML-KEM-768", backend)
    signer = get_signer("ML-DSA-65", backend)
    kem_pk, kem_sk = kem.generate_keypair()
    sig_pk, sig_sk = signer.generate_keypair()
    client_pk, client_sk = signer.generate_keypair()
    server = ServerIdentity(kem_pk, sig_pk)
    credentials = ClientCredentials("bench-client", client_sk)

    def handshake() -> tuple[bytes, bytes]:
        hello, pending = initiate(credentials, server, kem=kem, signer=signer)
        _, reply = accept(
            hello,
            kem=kem,
            signer=signer,
            server=server,
            kem_secret_key=kem_sk,
            signing_secret_key=sig_sk,
            client_public_key={"bench-client": client_pk}.get,
        )
        pending.finish(reply)
        return hello, reply

    hello, reply = handshake()
    return {
        "kem": kem.name,
        "signature": signer.name,
        "client_id_bytes": len(b"bench-client"),
        "hello_bytes": len(hello),
        "accept_bytes": len(reply),
        "data_frame_overhead_bytes": HEADER_SIZE + 16,
        "handshake": _timing(handshake, max(1, iterations // 4)),
    }


def _versions() -> dict[str, str | None]:
    found: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            found[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            found[package] = None
    if "liboqs" in available_backends():
        from qedge.crypto.backends.liboqs import version

        found["liboqs"] = version()
    return found


def main() -> None:
    """Run the benchmark and write the JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iterations", type=int, help="iterations per operation and backend")
    parser.add_argument("--output", type=Path, default=Path("results/summary/crypto.json"))
    parser.add_argument("--note", default="", help="free text stored with the results")
    args = parser.parse_args()

    backends = available_backends()
    results: dict[str, Any] = {}
    for backend in backends:
        iterations = args.iterations or DEFAULT_ITERATIONS[backend]
        results[backend] = {
            "kem": {name: _bench_kem(name, backend, iterations) for name in KEMS},
            "signature": {name: _bench_signature(name, backend, iterations) for name in SIGNATURES},
            "transport": _bench_transport(backend, iterations),
        }
    summary = {
        "generated_by": "bench/crypto.py",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "note": args.note,
        "machine": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
        },
        "packages": _versions(),
        "backends": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {args.output} for backends: {', '.join(backends)}")


if __name__ == "__main__":
    main()
