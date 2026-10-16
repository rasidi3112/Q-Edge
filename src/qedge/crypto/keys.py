"""Server key files written by ``qedge keys init`` and loaded at startup.

Key directory layout::

    keys.json     manifest: format, algorithm names, file names, SHA-256 fingerprints
    kem.pub       KEM public key
    kem.key       KEM secret key, mode 0600
    signing.pub   ML-DSA identity public key
    signing.key   ML-DSA identity secret key, mode 0600

Keys are stored as their raw FIPS 203/204 (or hybrid) encodings, so either
backend can load files written by the other. Loading refuses secret files that
group or others can read and checks that each key pair still matches.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qedge.crypto.backends import get_kem, get_signer
from qedge.crypto.errors import InvalidKeyError, KeyStoreError
from qedge.crypto.hybrid import HYBRID_X25519_ML_KEM_768
from qedge.crypto.kem import KEM_ALGORITHMS, Kem
from qedge.crypto.sign import ML_DSA_65, SIGNATURE_ALGORITHMS, Signer
from qedge.crypto.transport import ServerIdentity

MANIFEST = "keys.json"
FORMAT_VERSION = 1
DEFAULT_KEM = HYBRID_X25519_ML_KEM_768
DEFAULT_SIGNATURE = ML_DSA_65
SERVER_KEMS = (HYBRID_X25519_ML_KEM_768, *KEM_ALGORITHMS)
_FILES = {
    "kem_public_key": "kem.pub",
    "kem_secret_key": "kem.key",
    "signing_public_key": "signing.pub",
    "signing_secret_key": "signing.key",
}


@dataclass(frozen=True)
class ServerKeys:
    """The server's KEM key pair and ML-DSA identity key pair."""

    kem_algorithm: str
    signature_algorithm: str
    kem_public_key: bytes
    signing_public_key: bytes
    kem_secret_key: bytes = field(repr=False)
    signing_secret_key: bytes = field(repr=False)

    def identity(self) -> ServerIdentity:
        """Return the public keys that clients pin."""
        return ServerIdentity(self.kem_public_key, self.signing_public_key)


def fingerprint(public_key: bytes) -> str:
    """Hex SHA-256 of a public key, for out-of-band comparison."""
    return hashlib.sha256(public_key).hexdigest()


def _write(path: Path, data: bytes, secret: bool) -> None:
    path.unlink(missing_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600 if secret else 0o644)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)


def generate_server_keys(
    directory: Path, *, kem: Kem, signer: Signer, overwrite: bool = False
) -> ServerKeys:
    """Create a new key pair for each algorithm and write them to ``directory``.

    Raises:
        KeyStoreError: If key files already exist and ``overwrite`` is False,
            or ``kem`` is not one of `SERVER_KEMS`.
    """
    if kem.name not in SERVER_KEMS:
        raise KeyStoreError(f"KEM {kem.name!r} is not a server KEM; expected {SERVER_KEMS}")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    existing = [name for name in (MANIFEST, *_FILES.values()) if (directory / name).exists()]
    if existing and not overwrite:
        raise KeyStoreError(f"{directory} already holds {existing}; refusing to replace them")
    kem_public, kem_secret = kem.generate_keypair()
    signing_public, signing_secret = signer.generate_keypair()
    keys = ServerKeys(kem.name, signer.name, kem_public, signing_public, kem_secret, signing_secret)
    (directory / MANIFEST).unlink(missing_ok=True)
    for attribute, filename in _FILES.items():
        _write(
            directory / filename, getattr(keys, attribute), secret=attribute.endswith("secret_key")
        )
    manifest = {
        "format": FORMAT_VERSION,
        "kem": {"algorithm": kem.name, "fingerprint": fingerprint(kem_public)},
        "signature": {"algorithm": signer.name, "fingerprint": fingerprint(signing_public)},
        "files": _FILES,
    }
    # The manifest goes last, so an interrupted run leaves no loadable half-state.
    _write(directory / MANIFEST, json.dumps(manifest, indent=2).encode() + b"\n", secret=False)
    return keys


def _read(directory: Path, filename: str, secret: bool) -> bytes:
    path = directory / filename
    try:
        mode = path.stat().st_mode
        data = path.read_bytes()
    except OSError as exc:
        raise KeyStoreError(f"Cannot read {path}: {exc.strerror}") from exc
    if secret and os.name == "posix" and mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise KeyStoreError(f"{path} is accessible to group or others; run chmod 600")
    return data


def load_server_keys(directory: Path, *, backend: str) -> ServerKeys:
    """Load and check the keys in ``directory`` using ``backend``.

    Raises:
        KeyStoreError: If files are missing, unreadable, too permissive, have
            the wrong size or fingerprint, or a key pair does not match.
        BackendUnavailableError: If ``backend`` cannot provide the algorithms.
    """
    try:
        manifest: dict[str, Any] = json.loads(_read(directory, MANIFEST, secret=False))
        kem_algorithm = str(manifest["kem"]["algorithm"])
        signature_algorithm = str(manifest["signature"]["algorithm"])
        version = manifest["format"]
    except KeyStoreError as exc:
        raise KeyStoreError(f"{exc}; create keys with `qedge keys init --dir {directory}`") from exc
    except (ValueError, KeyError, TypeError) as exc:
        raise KeyStoreError(f"{directory / MANIFEST} is not a valid key manifest") from exc
    if version != FORMAT_VERSION:
        raise KeyStoreError(f"Unsupported key manifest format {version!r}")
    if kem_algorithm not in SERVER_KEMS or signature_algorithm not in SIGNATURE_ALGORITHMS:
        raise KeyStoreError(f"Unsupported algorithms {kem_algorithm!r}, {signature_algorithm!r}")
    kem = get_kem(kem_algorithm, backend)
    signer = get_signer(signature_algorithm, backend)

    raw = {
        attribute: _read(directory, filename, secret=attribute.endswith("secret_key"))
        for attribute, filename in _FILES.items()
    }
    expected = {
        "kem_public_key": kem.public_key_size,
        "kem_secret_key": kem.secret_key_size,
        "signing_public_key": signer.public_key_size,
        "signing_secret_key": signer.secret_key_size,
    }
    for attribute, size in expected.items():
        if len(raw[attribute]) != size:
            raise KeyStoreError(
                f"{_FILES[attribute]} is {len(raw[attribute])} bytes, expected {size}"
            )
    if fingerprint(raw["kem_public_key"]) != manifest["kem"].get("fingerprint") or fingerprint(
        raw["signing_public_key"]
    ) != manifest["signature"].get("fingerprint"):
        raise KeyStoreError("Public key files do not match the fingerprints in the manifest")

    keys = ServerKeys(kem_algorithm, signature_algorithm, **raw)
    ciphertext, shared_secret = kem.encapsulate(keys.kem_public_key)
    try:
        matches = kem.decapsulate(keys.kem_secret_key, ciphertext) == shared_secret
    except InvalidKeyError:
        matches = False
    if not matches:
        raise KeyStoreError("kem.key does not belong to kem.pub")
    probe = b"qedge key check"
    if not signer.verify(
        keys.signing_public_key, probe, signer.sign(keys.signing_secret_key, probe)
    ):
        raise KeyStoreError("signing.key does not belong to signing.pub")
    return keys
