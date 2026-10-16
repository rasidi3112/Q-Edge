"""ML-KEM and ML-DSA through liboqs (liboqs-python bindings).

``import oqs`` without a liboqs shared library makes liboqs-python clone and
build liboqs from GitHub, and exit the interpreter if that fails. This module
looks for the library in the same places first and reports the backend as
unavailable instead of importing. Set ``OQS_INSTALL_PATH`` to the prefix that
holds ``lib/liboqs.so`` (or ``.dylib``); the default is ``~/_oqs``.
"""

from __future__ import annotations

import ctypes.util
import importlib
import os
import platform
from functools import cache
from pathlib import Path
from typing import Any

from qedge.crypto.errors import BackendUnavailableError, InvalidCiphertextError, InvalidKeyError
from qedge.crypto.kem import KEM_ALGORITHMS, check_length
from qedge.crypto.sign import SIGNATURE_ALGORITHMS

NAME = "liboqs"


def _library_candidates() -> list[Path]:
    prefix = Path(os.environ.get("OQS_INSTALL_PATH", str(Path.home() / "_oqs")))
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    candidates = [prefix / sub / f"liboqs{suffix}" for sub in ("lib", "lib64")]
    for name in ("oqs", "liboqs"):
        found = ctypes.util.find_library(name)
        if found:
            candidates.append(Path(found))
    return candidates


@cache
def _oqs() -> Any:
    # find_library returns a bare soname on Linux, which the dynamic loader resolves.
    if not any(path.exists() or not path.is_absolute() for path in _library_candidates()):
        raise BackendUnavailableError(
            "liboqs shared library not found; build liboqs and set OQS_INSTALL_PATH"
        )
    try:
        return importlib.import_module("oqs")
    except ImportError as exc:
        raise BackendUnavailableError(
            "liboqs-python is not installed; install q-edge[pqc] for the liboqs backend"
        ) from exc


def version() -> str:
    """Return the liboqs library version.

    Raises:
        BackendUnavailableError: If liboqs or liboqs-python is missing.
    """
    return str(_oqs().oqs_version())


class LiboqsKem:
    """ML-KEM implemented by liboqs."""

    def __init__(self, algorithm: str) -> None:
        if algorithm not in KEM_ALGORITHMS:
            raise BackendUnavailableError(
                f"Unsupported KEM {algorithm!r}; expected {KEM_ALGORITHMS}"
            )
        oqs = _oqs()
        if algorithm not in oqs.get_enabled_kem_mechanisms():
            raise BackendUnavailableError(f"liboqs was built without {algorithm}")
        self.name = algorithm
        with oqs.KeyEncapsulation(algorithm) as kem:
            details = kem.details
        self.public_key_size = int(details["length_public_key"])
        self.secret_key_size = int(details["length_secret_key"])
        self.ciphertext_size = int(details["length_ciphertext"])
        self.shared_secret_size = int(details["length_shared_secret"])

    def generate_keypair(self) -> tuple[bytes, bytes]:
        """Return ``(public_key, secret_key)``."""
        with _oqs().KeyEncapsulation(self.name) as kem:
            public_key = kem.generate_keypair()
            return bytes(public_key), bytes(kem.export_secret_key())

    def encapsulate(self, public_key: bytes) -> tuple[bytes, bytes]:
        """Return ``(ciphertext, shared_secret)``."""
        check_length(public_key, self.public_key_size, "public key", InvalidKeyError)
        with _oqs().KeyEncapsulation(self.name) as kem:
            ciphertext, shared_secret = kem.encap_secret(public_key)
        return bytes(ciphertext), bytes(shared_secret)

    def decapsulate(self, secret_key: bytes, ciphertext: bytes) -> bytes:
        """Return the shared secret in ``ciphertext``."""
        check_length(secret_key, self.secret_key_size, "secret key", InvalidKeyError)
        check_length(ciphertext, self.ciphertext_size, "ciphertext", InvalidCiphertextError)
        with _oqs().KeyEncapsulation(self.name, secret_key=secret_key) as kem:
            return bytes(kem.decap_secret(ciphertext))


class LiboqsSigner:
    """ML-DSA implemented by liboqs."""

    def __init__(self, algorithm: str) -> None:
        if algorithm not in SIGNATURE_ALGORITHMS:
            raise BackendUnavailableError(
                f"Unsupported signature {algorithm!r}; expected {SIGNATURE_ALGORITHMS}"
            )
        oqs = _oqs()
        if algorithm not in oqs.get_enabled_sig_mechanisms():
            raise BackendUnavailableError(f"liboqs was built without {algorithm}")
        self.name = algorithm
        with oqs.Signature(algorithm) as sig:
            details = sig.details
        self.public_key_size = int(details["length_public_key"])
        self.secret_key_size = int(details["length_secret_key"])
        self.signature_size = int(details["length_signature"])

    def generate_keypair(self) -> tuple[bytes, bytes]:
        """Return ``(public_key, secret_key)``."""
        with _oqs().Signature(self.name) as sig:
            public_key = sig.generate_keypair()
            return bytes(public_key), bytes(sig.export_secret_key())

    def sign(self, secret_key: bytes, message: bytes) -> bytes:
        """Return an ML-DSA signature with an empty context."""
        check_length(secret_key, self.secret_key_size, "secret key", InvalidKeyError)
        with _oqs().Signature(self.name, secret_key=secret_key) as sig:
            return bytes(sig.sign(message))

    def verify(self, public_key: bytes, message: bytes, signature: bytes) -> bool:
        """Return True only for a valid signature."""
        check_length(public_key, self.public_key_size, "public key", InvalidKeyError)
        if len(signature) != self.signature_size:
            return False
        with _oqs().Signature(self.name) as sig:
            return bool(sig.verify(message, signature, public_key))
