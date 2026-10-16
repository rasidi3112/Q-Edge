"""Hybrid KEM: X25519 and ML-KEM-768 combined with HKDF-SHA256.

The shared secret stays secret as long as either X25519 or ML-KEM-768 is
unbroken, so recorded traffic is protected against a future quantum attacker
("harvest now, decrypt later") without trusting ML-KEM alone. See
docs/decisions/0005-hybrid-kem-at-nist-category-3.md.

Encodings (bytes):
    public key    = X25519 public (32) || ML-KEM-768 public (1184)
    secret key    = X25519 private (32) || ML-KEM-768 secret (2400) || public key
    ciphertext    = X25519 ephemeral public (32) || ML-KEM-768 ciphertext (1088)
    shared secret = HKDF-SHA256(ikm = ss_X25519 || ss_ML-KEM, salt = none,
                                info = label || public key || ciphertext), 32 bytes

Binding both public keys and both ciphertexts into ``info`` follows the
KEM-combiner analysis of Giacon, Heuer and Poettering ("KEM Combiners", 2018).
This is neither the TLS ``X25519MLKEM768`` group nor X-Wing and only
interoperates with itself.
"""

from __future__ import annotations

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from qedge.crypto.errors import InvalidCiphertextError, InvalidKeyError
from qedge.crypto.kem import ML_KEM_768, Kem, check_length

HYBRID_X25519_ML_KEM_768 = "X25519-ML-KEM-768"
_LABEL = b"qedge/hybrid-kem/X25519-ML-KEM-768/v1"
_X25519_SIZE = 32


def _raw_public(key: X25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


class HybridKem:
    """X25519 + ML-KEM-768 KEM; the ML-KEM half comes from any backend.

    Args:
        ml_kem: An ML-KEM-768 implementation.

    Raises:
        ValueError: If ``ml_kem`` is not ML-KEM-768.
    """

    def __init__(self, ml_kem: Kem) -> None:
        if ml_kem.name != ML_KEM_768:
            raise ValueError(f"HybridKem needs ML-KEM-768, got {ml_kem.name}")
        self._ml_kem = ml_kem
        self.name = HYBRID_X25519_ML_KEM_768
        self.public_key_size = _X25519_SIZE + ml_kem.public_key_size
        self.secret_key_size = _X25519_SIZE + ml_kem.secret_key_size + self.public_key_size
        self.ciphertext_size = _X25519_SIZE + ml_kem.ciphertext_size
        self.shared_secret_size = 32

    def generate_keypair(self) -> tuple[bytes, bytes]:
        """Return ``(public_key, secret_key)``."""
        x_private = X25519PrivateKey.generate()
        x_secret = x_private.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
        m_public, m_secret = self._ml_kem.generate_keypair()
        public_key = _raw_public(x_private) + m_public
        return public_key, x_secret + m_secret + public_key

    def encapsulate(self, public_key: bytes) -> tuple[bytes, bytes]:
        """Return ``(ciphertext, shared_secret)``."""
        check_length(public_key, self.public_key_size, "public key", InvalidKeyError)
        peer = X25519PublicKey.from_public_bytes(public_key[:_X25519_SIZE])
        ephemeral = X25519PrivateKey.generate()
        try:
            x_secret = ephemeral.exchange(peer)
        except ValueError as exc:  # low-order point: the exchange result is all zeros
            raise InvalidKeyError("X25519 public key is a low-order point") from exc
        m_ciphertext, m_secret = self._ml_kem.encapsulate(public_key[_X25519_SIZE:])
        ciphertext = _raw_public(ephemeral) + m_ciphertext
        return ciphertext, self._combine(x_secret, m_secret, public_key, ciphertext)

    def decapsulate(self, secret_key: bytes, ciphertext: bytes) -> bytes:
        """Return the shared secret in ``ciphertext``."""
        check_length(secret_key, self.secret_key_size, "secret key", InvalidKeyError)
        check_length(ciphertext, self.ciphertext_size, "ciphertext", InvalidCiphertextError)
        m_end = _X25519_SIZE + self._ml_kem.secret_key_size
        x_private = X25519PrivateKey.from_private_bytes(secret_key[:_X25519_SIZE])
        public_key = secret_key[m_end:]
        if _raw_public(x_private) != public_key[:_X25519_SIZE]:
            raise InvalidKeyError("Hybrid secret key does not match its embedded public key")
        ephemeral = X25519PublicKey.from_public_bytes(ciphertext[:_X25519_SIZE])
        try:
            x_secret = x_private.exchange(ephemeral)
        except ValueError as exc:
            raise InvalidCiphertextError("X25519 ciphertext is a low-order point") from exc
        m_secret = self._ml_kem.decapsulate(
            secret_key[_X25519_SIZE:m_end], ciphertext[_X25519_SIZE:]
        )
        return self._combine(x_secret, m_secret, public_key, ciphertext)

    @staticmethod
    def _combine(x_secret: bytes, m_secret: bytes, public_key: bytes, ciphertext: bytes) -> bytes:
        hkdf = HKDF(
            algorithm=hashes.SHA256(), length=32, salt=None, info=_LABEL + public_key + ciphertext
        )
        return hkdf.derive(x_secret + m_secret)
