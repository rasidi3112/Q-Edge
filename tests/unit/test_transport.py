import contextlib
from dataclasses import dataclass
from functools import cache

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from qedge.crypto.backends import get_kem, get_signer
from qedge.crypto.kem import Kem
from qedge.crypto.sign import Signer
from qedge.crypto.transport import (
    HEADER_SIZE,
    MAX_FRAME_SIZE,
    AuthenticationError,
    ClientCredentials,
    ExpiredFrameError,
    FrameError,
    FrameHeader,
    MessageType,
    PendingSession,
    ReplayError,
    ServerIdentity,
    Session,
    TransportError,
    accept,
    initiate,
    parse_header,
)


@dataclass(frozen=True)
class Parties:
    kem: Kem
    signer: Signer
    server: ServerIdentity
    kem_secret_key: bytes
    signing_secret_key: bytes
    client_public_key: bytes
    client_secret_key: bytes


@cache
def parties(backend: str = "pure", kem_name: str = "X25519-ML-KEM-768") -> Parties:
    kem, signer = get_kem(kem_name, backend), get_signer("ML-DSA-65", backend)
    kem_pk, kem_sk = kem.generate_keypair()
    sig_pk, sig_sk = signer.generate_keypair()
    client_pk, client_sk = signer.generate_keypair()
    return Parties(
        kem, signer, ServerIdentity(kem_pk, sig_pk), kem_sk, sig_sk, client_pk, client_sk
    )


def hello_for(p: Parties, client_id="hospital_A", clock=lambda: 1_000_000.0, server=None):
    return initiate(
        ClientCredentials(client_id, p.client_secret_key),
        server or p.server,
        kem=p.kem,
        signer=p.signer,
        clock=clock,
    )


def accept_for(p: Parties, hello: bytes, clock=lambda: 1_000_000.0, known=None):
    return accept(
        hello,
        kem=p.kem,
        signer=p.signer,
        server=p.server,
        kem_secret_key=p.kem_secret_key,
        signing_secret_key=p.signing_secret_key,
        client_public_key=(known if known is not None else {"hospital_A": p.client_public_key}).get,
        clock=clock,
    )


def established(p: Parties):
    hello, pending = hello_for(p)
    server, reply = accept_for(p, hello)
    return pending.finish(reply), server


def flip(frame: bytes, index: int) -> bytes:
    return frame[:index] + bytes([frame[index] ^ 1]) + frame[index + 1 :]


@pytest.mark.parametrize("kem_name", ["X25519-ML-KEM-768", "ML-KEM-768", "ML-KEM-1024"])
def test_handshake_and_data_in_both_directions(backend, kem_name):
    p = parties(backend, kem_name)
    hello, pending = hello_for(p)
    server, reply = accept_for(p, hello)
    client = pending.finish(reply)

    assert server.peer_id == "hospital_A"
    assert server.session_id == client.session_id
    for i in range(3):
        assert server.open(client.seal(b"update %d" % i)) == b"update %d" % i
        assert client.open(server.seal(b"model %d" % i)) == b"model %d" % i


def test_tampered_ciphertext_or_header_is_rejected():
    client, server = established(parties())
    frame = client.seal(b"update")

    with pytest.raises(AuthenticationError, match="AEAD"):
        server.open(flip(frame, len(frame) - 1))
    with pytest.raises(AuthenticationError, match="AEAD"):
        server.open(flip(frame, HEADER_SIZE - 1))  # timestamp is part of the AAD
    assert server.open(frame) == b"update"


def test_forged_frames_do_not_advance_the_replay_counter():
    client, server = established(parties())
    first, second = client.seal(b"a"), client.seal(b"b")
    forged = FrameHeader(1, MessageType.DATA, server.session_id, 5, 1_000_000).pack() + bytes(20)

    with pytest.raises(AuthenticationError):
        server.open(forged)
    assert server.open(first) == b"a"
    assert server.open(second) == b"b"
    with pytest.raises(ReplayError):
        server.open(first)


def test_frame_from_another_session_is_rejected():
    client_a, _ = established(parties())
    _, server_b = established(parties())

    with pytest.raises(AuthenticationError, match="another session"):
        server_b.open(client_a.seal(b"update"))


def test_old_or_future_timestamps_are_rejected():
    p = parties()
    hello, _ = hello_for(p, clock=lambda: 1_000_000.0)

    with pytest.raises(ExpiredFrameError):
        accept_for(p, hello, clock=lambda: 1_000_000.0 + 301)
    with pytest.raises(ExpiredFrameError):
        accept_for(p, hello, clock=lambda: 1_000_000.0 - 301)


def test_unknown_client_and_wrong_client_key_are_rejected():
    p = parties()
    hello, _ = hello_for(p)
    other_pk, _ = p.signer.generate_keypair()

    with pytest.raises(AuthenticationError, match="Unknown client"):
        accept_for(p, hello, known={})
    with pytest.raises(AuthenticationError, match="signature"):
        accept_for(p, hello, known={"hospital_A": other_pk})


def test_tampered_hello_fails_the_client_signature():
    p = parties()
    hello, _ = hello_for(p)

    with pytest.raises(AuthenticationError, match="signature"):
        accept_for(p, flip(hello, HEADER_SIZE + 2 + len(b"hospital_A") + 5))


def test_accept_from_an_impostor_server_key_is_rejected():
    p = parties()
    impostor_pk, _ = p.signer.generate_keypair()
    pinned = ServerIdentity(p.server.kem_public_key, impostor_pk)
    hello, pending = hello_for(p, server=pinned)
    # The real server derives a different transcript, so its ACCEPT cannot open.
    _, reply = accept_for(p, hello)

    with pytest.raises(AuthenticationError):
        pending.finish(reply)


def test_client_pinning_another_kem_key_cannot_complete_the_handshake():
    p = parties()
    other_kem_pk, _ = p.kem.generate_keypair()
    hello, pending = hello_for(p, server=ServerIdentity(other_kem_pk, p.server.signing_public_key))
    _, reply = accept(
        hello,
        kem=p.kem,
        signer=p.signer,
        server=ServerIdentity(other_kem_pk, p.server.signing_public_key),
        kem_secret_key=p.kem_secret_key,
        signing_secret_key=p.signing_secret_key,
        client_public_key={"hospital_A": p.client_public_key}.get,
        clock=lambda: 1_000_000.0,
    )

    with pytest.raises(AuthenticationError, match="AEAD"):
        pending.finish(reply)


def test_finish_can_only_run_once():
    p = parties()
    hello, pending = hello_for(p)
    _, reply = accept_for(p, hello)
    pending.finish(reply)

    with pytest.raises(TransportError, match="already finished"):
        pending.finish(reply)


def test_wrong_frame_types_are_rejected():
    p = parties()
    hello, pending = hello_for(p)
    _, reply = accept_for(p, hello)
    client = pending.finish(reply)

    with pytest.raises(FrameError, match="DATA"):
        client.open(reply)
    with pytest.raises(FrameError, match="HELLO"):
        accept_for(p, client.seal(b"x"))


def test_oversized_frames_and_plaintexts_are_rejected():
    client, server = established(parties())

    with pytest.raises(FrameError, match="exceeds"):
        server.open(b"\0" * (MAX_FRAME_SIZE + 1))
    with pytest.raises(TransportError, match="does not fit"):
        client.seal(b"\0" * MAX_FRAME_SIZE)


def test_client_id_length_is_bounded():
    p = parties()
    with pytest.raises(ValueError, match="Client id"):
        hello_for(p, client_id="")
    with pytest.raises(ValueError, match="Client id"):
        hello_for(p, client_id="x" * 257)


FUZZ = settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
header_like = st.builds(
    lambda body, t, c: FrameHeader(1, MessageType(t), b"\0" * 16, c, 1_000_000).pack() + body,
    st.binary(max_size=4000),
    st.sampled_from([1, 2, 3]),
    st.integers(0, 2**64 - 1),
)


@cache
def fuzz_sessions():
    return established(parties())


def fresh_pending(p: Parties) -> PendingSession:
    # Built directly so each example gets an unused handshake without a KEM call.
    session = Session(
        session_id=b"\0" * 16,
        peer_id="server",
        send_key=b"\1" * 32,
        recv_key=b"\2" * 32,
        clock=lambda: 1_000_000.0,
        max_clock_skew=300.0,
    )
    return PendingSession(session, b"transcript", p.server, p.signer)


@FUZZ
@given(st.one_of(st.binary(max_size=200), header_like))
def test_parsers_only_raise_transport_errors(data):
    p = parties()
    client, server = fuzz_sessions()
    for parse in (
        parse_header,
        server.open,
        client.open,
        fresh_pending(p).finish,
        lambda f: accept_for(p, f),
    ):
        with contextlib.suppress(TransportError):
            parse(data)
