"""Regression tests for the simulated cryptography found in the 99a5be6 audit."""

import builtins
import hashlib
import inspect
import pickle

import pytest

from qedge.crypto import transport
from qedge.crypto.backends import get_kem, get_signer
from qedge.crypto.transport import (
    HEADER_SIZE,
    ClientCredentials,
    ReplayError,
    ServerIdentity,
    accept,
    initiate,
)
from qedge.fl.weights import ModelSpec, TensorSpec, WeightFormatError, decode_weights


class _SetsBuiltinsFlag:
    def __reduce__(self):
        return (exec, ("import builtins; builtins._qedge_unpickled = True",))


def _handshake(backend, signature="ML-DSA-65"):
    kem = get_kem("X25519-ML-KEM-768", backend)
    signer = get_signer(signature, backend)
    server_kem_pk, server_kem_sk = kem.generate_keypair()
    server_sig_pk, server_sig_sk = signer.generate_keypair()
    client_pk, client_sk = signer.generate_keypair()
    server = ServerIdentity(server_kem_pk, server_sig_pk)
    hello, pending = initiate(
        ClientCredentials("hospital_A", client_sk), server, kem=kem, signer=signer
    )
    server_session, reply = accept(
        hello,
        kem=kem,
        signer=signer,
        server=server,
        kem_secret_key=server_kem_sk,
        signing_secret_key=server_sig_sk,
        client_public_key={"hospital_A": client_pk}.get,
    )
    return hello, pending.finish(reply), server_session, kem, signer


@pytest.mark.parametrize("algorithm", ["ML-DSA-65", "ML-DSA-87"])
def test_signature_verification_rejects_forged_signatures(backend, algorithm):
    signer = get_signer(algorithm, backend)
    public_key, _ = signer.generate_keypair()

    assert not signer.verify(public_key, b"evil", b"\0" * 32)
    assert not signer.verify(public_key, b"evil", b"\0" * signer.signature_size)


@pytest.mark.parametrize("algorithm", ["ML-KEM-768", "X25519-ML-KEM-768"])
def test_kem_shared_secret_cannot_be_derived_from_public_key(backend, algorithm):
    kem = get_kem(algorithm, backend)
    public_key, secret_key = kem.generate_keypair()
    _, attacker_secret_key = kem.generate_keypair()

    ciphertext, shared_secret = kem.encapsulate(public_key)

    assert shared_secret != hashlib.sha256(public_key + b"simulated_encap").digest()
    assert shared_secret != hashlib.sha256(public_key).digest()
    assert kem.decapsulate(attacker_secret_key, ciphertext) != shared_secret
    assert kem.decapsulate(secret_key, ciphertext) == shared_secret


def test_received_weights_are_never_unpickled(backend):
    _, client, server, _, _ = _handshake(backend)
    spec = ModelSpec((TensorSpec("w", (2,), "float32"),))
    frame = client.seal(pickle.dumps(_SetsBuiltinsFlag()))

    with pytest.raises(WeightFormatError):
        decode_weights(spec, server.open(frame))
    executed = getattr(builtins, "_qedge_unpickled", False)
    if executed:
        del builtins._qedge_unpickled

    assert not executed


def test_transport_does_not_default_to_simulated_crypto():
    for function in (transport.initiate, transport.accept):
        parameters = inspect.signature(function).parameters
        assert parameters["kem"].default is inspect.Parameter.empty
        assert parameters["signer"].default is inspect.Parameter.empty
    assert "simulat" not in inspect.getsource(transport).lower()


def test_handshake_carries_the_kem_ciphertext_to_the_server(backend):
    hello, client, server, kem, _ = _handshake(backend)

    assert len(hello) > HEADER_SIZE + kem.ciphertext_size
    assert server.open(client.seal(b"round 1 update")) == b"round 1 update"
    assert client.open(server.seal(b"round 2 model")) == b"round 2 model"


def test_replayed_frame_is_rejected(backend):
    _, client, server, _, _ = _handshake(backend)
    frame = client.seal(b"update")
    server.open(frame)

    with pytest.raises(ReplayError):
        server.open(frame)


@pytest.mark.parametrize(("signature", "size"), [("ML-DSA-65", 3309), ("ML-DSA-87", 4627)])
def test_frame_signature_length_comes_from_the_signature_scheme(signature, size):
    hello, client, server, kem, signer = _handshake("pure", signature)

    assert signer.signature_size == size
    assert len(hello) == HEADER_SIZE + 2 + len(b"hospital_A") + kem.ciphertext_size + size
    assert server.open(client.seal(b"update")) == b"update"
