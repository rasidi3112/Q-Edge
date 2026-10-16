"""Pure-Python ML-KEM and ML-DSA from kyber-py and dilithium-py.

Not constant-time, tests only. These packages implement FIPS 203 and FIPS 204
and interoperate with liboqs, but their arithmetic leaks timing, so a server
must not use them for keys that matter. Install with ``pip install
'q-edge[pqc-pure]'``.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

from qedge.crypto.errors import BackendUnavailableError, InvalidCiphertextError, InvalidKeyError
from qedge.crypto.kem import KEM_ALGORITHMS, check_length
from qedge.crypto.sign import SIGNATURE_ALGORITHMS

logger = logging.getLogger(__name__)

NAME = "pure"
_warned = False

# FIPS 203 / FIPS 204 encoding sizes (public key, secret key, ciphertext or signature).
_KEM_SIZES = {"ML-KEM-768": (1184, 2400, 1088), "ML-KEM-1024": (1568, 3168, 1568)}
_SIG_SIZES = {"ML-DSA-65": (1952, 4032, 3309), "ML-DSA-87": (2592, 4896, 4627)}


def _warn_once() -> None:
    global _warned
    if not _warned:
        logger.warning("Using the pure-Python ML-KEM/ML-DSA backend: not constant-time, tests only")
        _warned = True


def _load(module: str, attribute: str) -> Any:
    try:
        imported = importlib.import_module(module)
    except ImportError as exc:
        raise BackendUnavailableError(
            f"{module} is not installed; install q-edge[pqc-pure] for the pure backend"
        ) from exc
    return getattr(imported, attribute)


class PureKem:
    """ML-KEM via kyber-py (not constant-time, tests only)."""

    def __init__(self, algorithm: str) -> None:
        if algorithm not in KEM_ALGORITHMS:
            raise BackendUnavailableError(
                f"Unsupported KEM {algorithm!r}; expected {KEM_ALGORITHMS}"
            )
        _warn_once()
        self.name = algorithm
        self._impl = _load("kyber_py.ml_kem", algorithm.replace("-", "_"))
        self.public_key_size, self.secret_key_size, self.ciphertext_size = _KEM_SIZES[algorithm]
        self.shared_secret_size = 32

    def generate_keypair(self) -> tuple[bytes, bytes]:
        """Return ``(public_key, secret_key)``."""
        public_key, secret_key = self._impl.keygen()
        return bytes(public_key), bytes(secret_key)

    def encapsulate(self, public_key: bytes) -> tuple[bytes, bytes]:
        """Return ``(ciphertext, shared_secret)``."""
        check_length(public_key, self.public_key_size, "public key", InvalidKeyError)
        try:
            shared_secret, ciphertext = self._impl.encaps(public_key)
        except ValueError as exc:
            raise InvalidKeyError(f"{self.name} public key failed validation") from exc
        return bytes(ciphertext), bytes(shared_secret)

    def decapsulate(self, secret_key: bytes, ciphertext: bytes) -> bytes:
        """Return the shared secret in ``ciphertext``."""
        check_length(secret_key, self.secret_key_size, "secret key", InvalidKeyError)
        check_length(ciphertext, self.ciphertext_size, "ciphertext", InvalidCiphertextError)
        try:
            return bytes(self._impl.decaps(secret_key, ciphertext))
        except ValueError as exc:
            raise InvalidKeyError(f"{self.name} secret key failed validation") from exc


class PureSigner:
    """ML-DSA via dilithium-py (not constant-time, tests only)."""

    def __init__(self, algorithm: str) -> None:
        if algorithm not in SIGNATURE_ALGORITHMS:
            raise BackendUnavailableError(
                f"Unsupported signature {algorithm!r}; expected {SIGNATURE_ALGORITHMS}"
            )
        _warn_once()
        self.name = algorithm
        self._impl = _load("dilithium_py.ml_dsa", algorithm.replace("-", "_"))
        self.public_key_size, self.secret_key_size, self.signature_size = _SIG_SIZES[algorithm]

    def generate_keypair(self) -> tuple[bytes, bytes]:
        """Return ``(public_key, secret_key)``."""
        public_key, secret_key = self._impl.keygen()
        return bytes(public_key), bytes(secret_key)

    def sign(self, secret_key: bytes, message: bytes) -> bytes:
        """Return a hedged (randomized) ML-DSA signature with an empty context."""
        check_length(secret_key, self.secret_key_size, "secret key", InvalidKeyError)
        return bytes(self._impl.sign(secret_key, message))

    def verify(self, public_key: bytes, message: bytes, signature: bytes) -> bool:
        """Return True only for a valid signature."""
        check_length(public_key, self.public_key_size, "public key", InvalidKeyError)
        if len(signature) != self.signature_size:
            return False
        return bool(self._impl.verify(public_key, message, signature))
