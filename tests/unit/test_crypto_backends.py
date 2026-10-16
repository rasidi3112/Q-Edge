import sys

import pytest

from qedge.crypto.backends import available_backends, get_kem, get_signer
from qedge.crypto.backends import liboqs as liboqs_backend
from qedge.crypto.errors import BackendUnavailableError, InvalidCiphertextError, InvalidKeyError

# (public key, secret key, ciphertext or signature) sizes from FIPS 203 and FIPS 204.
FIPS_KEM_SIZES = {"ML-KEM-768": (1184, 2400, 1088), "ML-KEM-1024": (1568, 3168, 1568)}
FIPS_SIG_SIZES = {"ML-DSA-65": (1952, 4032, 3309), "ML-DSA-87": (2592, 4896, 4627)}
KEMS = [*FIPS_KEM_SIZES, "X25519-ML-KEM-768"]


@pytest.mark.parametrize("algorithm", KEMS)
def test_kem_roundtrip_produces_equal_secrets(backend, algorithm):
    kem = get_kem(algorithm, backend)
    public_key, secret_key = kem.generate_keypair()

    ciphertext, shared_secret = kem.encapsulate(public_key)

    assert len(shared_secret) == kem.shared_secret_size == 32
    assert kem.decapsulate(secret_key, ciphertext) == shared_secret


@pytest.mark.parametrize("algorithm", FIPS_KEM_SIZES)
def test_kem_encodings_have_fips_203_sizes(backend, algorithm):
    kem = get_kem(algorithm, backend)
    public_key, secret_key = kem.generate_keypair()
    ciphertext, _ = kem.encapsulate(public_key)

    assert (len(public_key), len(secret_key), len(ciphertext)) == FIPS_KEM_SIZES[algorithm]
    assert (kem.public_key_size, kem.secret_key_size, kem.ciphertext_size) == FIPS_KEM_SIZES[
        algorithm
    ]


@pytest.mark.parametrize("algorithm", KEMS)
def test_decapsulating_with_another_key_gives_an_unrelated_secret(backend, algorithm):
    kem = get_kem(algorithm, backend)
    public_key, _ = kem.generate_keypair()
    _, other_secret_key = kem.generate_keypair()
    ciphertext, shared_secret = kem.encapsulate(public_key)

    assert kem.decapsulate(other_secret_key, ciphertext) != shared_secret


@pytest.mark.parametrize("algorithm", KEMS)
def test_encapsulation_is_randomized(backend, algorithm):
    kem = get_kem(algorithm, backend)
    public_key, _ = kem.generate_keypair()

    assert kem.encapsulate(public_key) != kem.encapsulate(public_key)


@pytest.mark.parametrize("algorithm", KEMS)
def test_kem_rejects_inputs_of_the_wrong_length(backend, algorithm):
    kem = get_kem(algorithm, backend)
    public_key, secret_key = kem.generate_keypair()
    ciphertext, _ = kem.encapsulate(public_key)

    with pytest.raises(InvalidKeyError):
        kem.encapsulate(public_key[:-1])
    with pytest.raises(InvalidKeyError):
        kem.decapsulate(secret_key + b"\0", ciphertext)
    with pytest.raises(InvalidCiphertextError):
        kem.decapsulate(secret_key, ciphertext[:-1])


@pytest.mark.parametrize("algorithm", FIPS_SIG_SIZES)
def test_signature_roundtrip_and_fips_204_sizes(backend, algorithm):
    signer = get_signer(algorithm, backend)
    public_key, secret_key = signer.generate_keypair()

    signature = signer.sign(secret_key, b"round 3 update")

    assert (len(public_key), len(secret_key), len(signature)) == FIPS_SIG_SIZES[algorithm]
    assert signer.signature_size == len(signature)
    assert signer.verify(public_key, b"round 3 update", signature)


@pytest.mark.parametrize("algorithm", FIPS_SIG_SIZES)
def test_verify_rejects_changed_messages_and_signatures(backend, algorithm):
    signer = get_signer(algorithm, backend)
    public_key, secret_key = signer.generate_keypair()
    signature = signer.sign(secret_key, b"update")
    flipped = bytes([signature[0] ^ 1]) + signature[1:]

    assert not signer.verify(public_key, b"updatf", signature)
    assert not signer.verify(public_key, b"update", flipped)
    assert not signer.verify(public_key, b"update", signature[:-1])
    assert not signer.verify(public_key, b"update", b"\0" * 32)
    with pytest.raises(InvalidKeyError):
        signer.verify(public_key[:-1], b"update", signature)


def test_unknown_backend_and_algorithm_raise():
    with pytest.raises(BackendUnavailableError, match="backend"):
        get_kem("ML-KEM-768", "simulated")
    with pytest.raises(BackendUnavailableError, match="Unsupported"):
        get_kem("Kyber1024", "pure")
    with pytest.raises(BackendUnavailableError, match="Unsupported"):
        get_signer("Dilithium5", "pure")


def test_liboqs_backend_reports_a_missing_library_without_importing_oqs(monkeypatch, tmp_path):
    was_imported = "oqs" in sys.modules
    monkeypatch.setenv("OQS_INSTALL_PATH", str(tmp_path))
    monkeypatch.setattr(liboqs_backend.ctypes.util, "find_library", lambda name: None)
    liboqs_backend._oqs.cache_clear()
    try:
        with pytest.raises(BackendUnavailableError, match="shared library"):
            get_kem("ML-KEM-768", "liboqs")
    finally:
        liboqs_backend._oqs.cache_clear()

    assert ("oqs" in sys.modules) == was_imported


@pytest.mark.skipif("liboqs" not in available_backends(), reason="liboqs not installed")
def test_liboqs_and_pure_backends_interoperate():
    for producer, consumer in (("liboqs", "pure"), ("pure", "liboqs")):
        kem_a, kem_b = get_kem("ML-KEM-768", producer), get_kem("ML-KEM-768", consumer)
        public_key, secret_key = kem_a.generate_keypair()
        ciphertext, shared_secret = kem_b.encapsulate(public_key)
        assert kem_a.decapsulate(secret_key, ciphertext) == shared_secret

        sig_a, sig_b = get_signer("ML-DSA-65", producer), get_signer("ML-DSA-65", consumer)
        public_key, secret_key = sig_a.generate_keypair()
        assert sig_b.verify(public_key, b"m", sig_a.sign(secret_key, b"m"))
