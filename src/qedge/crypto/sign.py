"""Signature interface shared by all backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

ML_DSA_65 = "ML-DSA-65"
ML_DSA_87 = "ML-DSA-87"
SIGNATURE_ALGORITHMS = (ML_DSA_65, ML_DSA_87)


@runtime_checkable
class Signer(Protocol):
    """A signature scheme with fixed-size byte encodings.

    Attributes:
        name: Algorithm name, for example ``"ML-DSA-65"``.
        public_key_size: Verification key length in bytes.
        secret_key_size: Signing key length in bytes.
        signature_size: Signature length in bytes; ML-DSA signatures have a
            fixed length per parameter set.
    """

    name: str
    public_key_size: int
    secret_key_size: int
    signature_size: int

    def generate_keypair(self) -> tuple[bytes, bytes]:
        """Return a fresh ``(public_key, secret_key)`` pair."""
        ...

    def sign(self, secret_key: bytes, message: bytes) -> bytes:
        """Return a ``signature_size``-byte signature over ``message``.

        Raises:
            InvalidKeyError: If ``secret_key`` has the wrong length.
        """
        ...

    def verify(self, public_key: bytes, message: bytes, signature: bytes) -> bool:
        """Return True only for a valid signature; a wrong-length signature is False.

        Raises:
            InvalidKeyError: If ``public_key`` has the wrong length.
        """
        ...
