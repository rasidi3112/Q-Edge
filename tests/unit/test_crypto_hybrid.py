import pytest

from qedge.crypto.backends import get_kem
from qedge.crypto.errors import InvalidCiphertextError, InvalidKeyError
from qedge.crypto.hybrid import HybridKem


@pytest.fixture
def hybrid(backend):
    return get_kem("X25519-ML-KEM-768", backend)


def test_hybrid_sizes_are_x25519_plus_ml_kem_768(hybrid):
    assert hybrid.public_key_size == 32 + 1184
    assert hybrid.ciphertext_size == 32 + 1088
    assert hybrid.secret_key_size == 32 + 2400 + hybrid.public_key_size


def test_changing_either_ciphertext_half_changes_the_secret(hybrid):
    public_key, secret_key = hybrid.generate_keypair()
    ciphertext, shared_secret = hybrid.encapsulate(public_key)
    x25519_flipped = bytes([ciphertext[0] ^ 1]) + ciphertext[1:]
    ml_kem_flipped = ciphertext[:40] + bytes([ciphertext[40] ^ 1]) + ciphertext[41:]

    assert hybrid.decapsulate(secret_key, x25519_flipped) != shared_secret
    assert hybrid.decapsulate(secret_key, ml_kem_flipped) != shared_secret


def test_hybrid_rejects_low_order_x25519_points(hybrid):
    public_key, secret_key = hybrid.generate_keypair()
    ciphertext, _ = hybrid.encapsulate(public_key)

    with pytest.raises(InvalidKeyError, match="low-order"):
        hybrid.encapsulate(bytes(32) + public_key[32:])
    with pytest.raises(InvalidCiphertextError, match="low-order"):
        hybrid.decapsulate(secret_key, bytes(32) + ciphertext[32:])


def test_hybrid_rejects_a_secret_key_with_a_foreign_public_key(hybrid):
    public_key, secret_key = hybrid.generate_keypair()
    other_public, _ = hybrid.generate_keypair()
    ciphertext, _ = hybrid.encapsulate(public_key)
    mismatched = secret_key[: -hybrid.public_key_size] + other_public

    with pytest.raises(InvalidKeyError, match="embedded public key"):
        hybrid.decapsulate(mismatched, ciphertext)


def test_hybrid_requires_ml_kem_768():
    with pytest.raises(ValueError, match="ML-KEM-768"):
        HybridKem(get_kem("ML-KEM-1024", "pure"))
