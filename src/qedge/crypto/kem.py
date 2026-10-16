"""Key encapsulation interface shared by all backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

ML_KEM_768 = "ML-KEM-768"
ML_KEM_1024 = "ML-KEM-1024"
KEM_ALGORITHMS = (ML_KEM_768, ML_KEM_1024)


@runtime_checkable
class Kem(Protocol):
    """A key encapsulation mechanism with fixed-size byte encodings.

    Attributes:
        name: Algorithm name, for example ``"ML-KEM-768"``.
        public_key_size: Encapsulation key length in bytes.
        secret_key_size: Decapsulation key length in bytes.
        ciphertext_size: Ciphertext length in bytes.
        shared_secret_size: Shared secret length in bytes.
    """

    name: str
    public_key_size: int
    secret_key_size: int
    ciphertext_size: int
    shared_secret_size: int

    def generate_keypair(self) -> tuple[bytes, bytes]:
        """Return a fresh ``(public_key, secret_key)`` pair."""
        ...

    def encapsulate(self, public_key: bytes) -> tuple[bytes, bytes]:
        """Return ``(ciphertext, shared_secret)`` for ``public_key``.

        Raises:
            InvalidKeyError: If ``public_key`` has the wrong length or encoding.
        """
        ...

    def decapsulate(self, secret_key: bytes, ciphertext: bytes) -> bytes:
        """Return the shared secret encapsulated in ``ciphertext``.

        ML-KEM uses implicit rejection: a well-formed but wrong ciphertext yields
        an unrelated secret instead of an error.

        Raises:
            InvalidKeyError: If ``secret_key`` has the wrong length.
            InvalidCiphertextError: If ``ciphertext`` has the wrong length or encoding.
        """
        ...


def check_length(value: bytes, expected: int, what: str, error: type[ValueError]) -> None:
    """Raise ``error`` unless ``value`` is exactly ``expected`` bytes long."""
    if len(value) != expected:
        raise error(f"{what} must be {expected} bytes, got {len(value)}")
