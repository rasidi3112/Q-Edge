"""Regression tests for the simulated cryptography found in the 99a5be6 audit."""

import asyncio
import builtins
import hashlib
import hmac
import pickle
import zlib

import pytest

from qedge.crypto.transport import PQCTransportLayer
from qedge.server.auth import PQCAlgorithm, PQCProvider

SIMULATED = pytest.mark.xfail(strict=True, reason="simulated cryptography")


class _SetsBuiltinsFlag:
    def __reduce__(self):
        return (exec, ("import builtins; builtins._qedge_unpickled = True",))


def _session(transport: PQCTransportLayer) -> None:
    asyncio.run(transport.establish_session(b"server-public-key"))


@SIMULATED
def test_signature_verification_rejects_forged_signatures():
    provider = PQCProvider()

    assert not provider.verify(b"evil", b"\0" * 32, b"", PQCAlgorithm.DILITHIUM_5)


@SIMULATED
def test_kem_shared_secret_cannot_be_derived_from_public_key():
    provider = PQCProvider()
    keypair = provider.generate_keypair(PQCAlgorithm.KYBER_768)

    shared_secret = provider.encapsulate(keypair.public_key, PQCAlgorithm.KYBER_768).shared_secret

    assert shared_secret != hashlib.sha256(keypair.public_key + b"simulated_encap").digest()


@SIMULATED
def test_received_weights_are_never_unpickled():
    transport = PQCTransportLayer("hospital_A")
    _session(transport)
    frame = transport._encrypt(
        zlib.compress(pickle.dumps(_SetsBuiltinsFlag())), context=b"weight_encryption"
    )

    transport.decrypt_weights(frame)
    executed = getattr(builtins, "_qedge_unpickled", False)
    if executed:
        del builtins._qedge_unpickled

    assert not executed


@SIMULATED
def test_transport_does_not_default_to_simulated_crypto():
    assert PQCTransportLayer("hospital_A").use_simulation is False


@SIMULATED
def test_handshake_carries_the_kem_ciphertext_to_the_server():
    transport = PQCTransportLayer("hospital_A")
    _session(transport)

    assert getattr(transport.session, "kem_ciphertext", b"")


@SIMULATED
def test_replayed_frame_is_rejected():
    transport = PQCTransportLayer("hospital_A")
    _session(transport)
    frame = transport.create_authenticated_message(b"update")
    transport.parse_authenticated_message(frame)

    with pytest.raises(ValueError):
        transport.parse_authenticated_message(frame)


@SIMULATED
def test_frame_signature_length_comes_from_the_signature_scheme():
    transport = PQCTransportLayer("hospital_A")
    _session(transport)
    ml_dsa_65_signature_size = 3309
    transport.sign = lambda data: (hmac.new(b"k", data, hashlib.sha512).digest() * 52)[
        :ml_dsa_65_signature_size
    ]
    transport.verify = lambda data, signature, public_key: (
        len(signature) == ml_dsa_65_signature_size
    )
    frame = transport.create_authenticated_message(b"update")

    payload, _ = transport.parse_authenticated_message(frame)

    assert payload == b"update"
