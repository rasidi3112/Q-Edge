"""Backend registry: every caller names its backend; nothing falls back silently.

``liboqs`` is the backend for anything that should be secure. ``pure`` is
pure Python, not constant-time, and meant for tests and CI only.
"""

from __future__ import annotations

from qedge.crypto.errors import BackendUnavailableError
from qedge.crypto.kem import Kem
from qedge.crypto.sign import Signer

BACKENDS = ("liboqs", "pure")


def get_kem(algorithm: str, backend: str) -> Kem:
    """Return the KEM ``algorithm`` from ``backend``.

    Raises:
        BackendUnavailableError: If the backend is unknown, not installed, or
            lacks the algorithm.
    """
    if backend == "liboqs":
        from qedge.crypto.backends.liboqs import LiboqsKem

        return LiboqsKem(algorithm)
    if backend == "pure":
        from qedge.crypto.backends.pure import PureKem

        return PureKem(algorithm)
    raise BackendUnavailableError(f"Unknown crypto backend {backend!r}; expected {BACKENDS}")


def get_signer(algorithm: str, backend: str) -> Signer:
    """Return the signature scheme ``algorithm`` from ``backend``.

    Raises:
        BackendUnavailableError: If the backend is unknown, not installed, or
            lacks the algorithm.
    """
    if backend == "liboqs":
        from qedge.crypto.backends.liboqs import LiboqsSigner

        return LiboqsSigner(algorithm)
    if backend == "pure":
        from qedge.crypto.backends.pure import PureSigner

        return PureSigner(algorithm)
    raise BackendUnavailableError(f"Unknown crypto backend {backend!r}; expected {BACKENDS}")


def available_backends() -> list[str]:
    """Return the backends that can provide ML-KEM-768 and ML-DSA-65 here."""
    found = []
    for backend in BACKENDS:
        try:
            get_kem("ML-KEM-768", backend)
            get_signer("ML-DSA-65", backend)
        except BackendUnavailableError:
            continue
        found.append(backend)
    return found
