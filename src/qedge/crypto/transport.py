"""Authenticated, encrypted channel between a client and the server.

Handshake, one round trip:

1. The client encapsulates to the server's static KEM public key, picks a
   random 16-byte session id and sends HELLO, signed with its ML-DSA key.
2. The server verifies that signature against the client's registered key,
   decapsulates, derives the session keys and answers with ACCEPT: an AEAD
   frame under the new server-to-client key whose plaintext is the server's
   ML-DSA signature over the transcript hash. Opening it shows the server holds
   the KEM secret key; the signature shows it holds the pinned identity key.

Data frames use ChaCha20-Poly1305 with one key per direction. Both keys come
from HKDF-SHA256 over the KEM shared secret, salted with the session id and
bound to the transcript (algorithm names, server public keys, signed HELLO).

Frame layout, big-endian::

    header = "QE" | version u8 | type u8 | session id (16) | counter u64 | timestamp u64
    HELLO  = header | client id length u16 | client id (UTF-8) | KEM ciphertext | signature
    ACCEPT = DATA = header | AEAD ciphertext and tag;  AAD = header,
             nonce = 4 zero bytes || counter

KEM ciphertext and signature lengths come from the algorithms in use. A
receiver rejects a frame whose counter is not above the last accepted one,
whose session id differs, or whose timestamp is more than ``max_clock_skew``
seconds from its clock. `accept` keeps no state: the caller must refuse a HELLO
session id it has already accepted. `Session` is not thread-safe.
"""

from __future__ import annotations

import hashlib
import secrets
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from qedge.crypto.errors import InvalidCiphertextError
from qedge.crypto.kem import Kem
from qedge.crypto.sign import Signer

VERSION = 1
SESSION_ID_SIZE = 16
MAX_FRAME_SIZE = 64 * 1024 * 1024
MAX_CLIENT_ID_BYTES = 256
DEFAULT_MAX_CLOCK_SKEW = 300.0

_MAGIC = b"QE"
_HEADER = struct.Struct(">2sBB16sQQ")
HEADER_SIZE = _HEADER.size
_TAG_SIZE = 16
_MAX_COUNTER = 2**64 - 1
_LABEL = b"qedge/transport/v1"
_ACCEPT_LABEL = _LABEL + b"/accept"

Clock = Callable[[], float]


class MessageType(IntEnum):
    """Frame type carried in the header."""

    HELLO = 1
    ACCEPT = 2
    DATA = 3


class TransportError(ValueError):
    """Base class for every frame the transport refuses."""


class FrameError(TransportError):
    """The frame is malformed: wrong magic, version, type, length or encoding."""


class AuthenticationError(TransportError):
    """A signature, AEAD tag, client id or session id did not check out."""


class ReplayError(TransportError):
    """The frame counter is not above the last accepted counter."""


class ExpiredFrameError(TransportError):
    """The frame timestamp is outside the allowed clock skew."""


@dataclass(frozen=True)
class FrameHeader:
    """Parsed frame header; every field is authenticated as AAD or signed."""

    version: int
    msg_type: MessageType
    session_id: bytes
    counter: int
    timestamp: int

    def pack(self) -> bytes:
        """Encode the header as ``HEADER_SIZE`` bytes."""
        return _HEADER.pack(
            _MAGIC, self.version, self.msg_type, self.session_id, self.counter, self.timestamp
        )


def parse_header(frame: bytes) -> FrameHeader:
    """Parse and check the fixed header at the start of ``frame``.

    Raises:
        FrameError: If the frame is shorter than a header, larger than
            ``MAX_FRAME_SIZE``, or has an unknown magic, version or type.
    """
    if len(frame) < HEADER_SIZE:
        raise FrameError(f"Frame of {len(frame)} bytes is shorter than the header")
    if len(frame) > MAX_FRAME_SIZE:
        raise FrameError(f"Frame of {len(frame)} bytes exceeds {MAX_FRAME_SIZE}")
    magic, version, msg_type, session_id, counter, timestamp = _HEADER.unpack_from(frame)
    if magic != _MAGIC:
        raise FrameError("Bad frame magic")
    if version != VERSION:
        raise FrameError(f"Unsupported frame version {version}")
    try:
        parsed_type = MessageType(msg_type)
    except ValueError as exc:
        raise FrameError(f"Unknown frame type {msg_type}") from exc
    return FrameHeader(version, parsed_type, session_id, counter, timestamp)


@dataclass(frozen=True)
class ServerIdentity:
    """Server public keys that a client pins."""

    kem_public_key: bytes
    signing_public_key: bytes


@dataclass(frozen=True)
class ClientCredentials:
    """Client id and the ML-DSA secret key registered for it."""

    client_id: str
    signing_secret_key: bytes


def _check_timestamp(timestamp: int, clock: Clock, max_clock_skew: float) -> None:
    if abs(clock() - timestamp) > max_clock_skew:
        raise ExpiredFrameError(f"Frame timestamp {timestamp} is outside the allowed skew")


def _nonce(counter: int) -> bytes:
    return b"\0\0\0\0" + counter.to_bytes(8, "big")


class Session:
    """An established channel; create it with `initiate` and `accept`."""

    def __init__(
        self,
        *,
        session_id: bytes,
        peer_id: str,
        send_key: bytes,
        recv_key: bytes,
        clock: Clock,
        max_clock_skew: float,
    ) -> None:
        self.session_id = session_id
        self.peer_id = peer_id
        self._send = ChaCha20Poly1305(send_key)
        self._recv = ChaCha20Poly1305(recv_key)
        self._send_counter = 0
        self._recv_next = 0
        self._clock = clock
        self._max_clock_skew = max_clock_skew

    def seal(self, plaintext: bytes) -> bytes:
        """Encrypt ``plaintext`` into a DATA frame.

        Raises:
            TransportError: If the frame would exceed ``MAX_FRAME_SIZE`` or the
                send counter is exhausted.
        """
        return self._seal(MessageType.DATA, plaintext)

    def open(self, frame: bytes) -> bytes:
        """Authenticate and decrypt a DATA frame from the peer.

        Raises:
            FrameError: If the frame is malformed or not a DATA frame.
            AuthenticationError: If the session id or AEAD tag does not match.
            ReplayError: If the counter was already used or is older.
            ExpiredFrameError: If the timestamp is outside the allowed skew.
        """
        return self._open(frame, MessageType.DATA)

    def _seal(self, msg_type: MessageType, plaintext: bytes) -> bytes:
        if HEADER_SIZE + len(plaintext) + _TAG_SIZE > MAX_FRAME_SIZE:
            raise TransportError(f"Plaintext of {len(plaintext)} bytes does not fit in a frame")
        if self._send_counter > _MAX_COUNTER:
            raise TransportError("Send counter exhausted; start a new session")
        header = FrameHeader(
            VERSION, msg_type, self.session_id, self._send_counter, int(self._clock())
        ).pack()
        ciphertext = self._send.encrypt(_nonce(self._send_counter), plaintext, header)
        self._send_counter += 1
        return header + ciphertext

    def _open(self, frame: bytes, expected: MessageType) -> bytes:
        header = parse_header(frame)
        if header.msg_type != expected:
            raise FrameError(f"Expected a {expected.name} frame, got {header.msg_type.name}")
        if len(frame) < HEADER_SIZE + _TAG_SIZE:
            raise FrameError("Frame is too short to hold an AEAD tag")
        if header.session_id != self.session_id:
            raise AuthenticationError("Frame belongs to another session")
        if header.counter < self._recv_next:
            raise ReplayError(f"Counter {header.counter} was already used or is older")
        _check_timestamp(header.timestamp, self._clock, self._max_clock_skew)
        try:
            plaintext = self._recv.decrypt(
                _nonce(header.counter), frame[HEADER_SIZE:], frame[:HEADER_SIZE]
            )
        except InvalidTag as exc:
            raise AuthenticationError("AEAD tag does not verify") from exc
        # Advance only after authentication, so forged frames cannot burn counters.
        self._recv_next = header.counter + 1
        return plaintext


def _transcript(kem: Kem, signer: Signer, server: ServerIdentity, signed_hello: bytes) -> bytes:
    digest = hashlib.sha256()
    for part in (
        _LABEL,
        kem.name.encode(),
        signer.name.encode(),
        server.kem_public_key,
        server.signing_public_key,
        signed_hello,
    ):
        digest.update(len(part).to_bytes(4, "big"))
        digest.update(part)
    return digest.digest()


def _session_keys(
    shared_secret: bytes, session_id: bytes, transcript: bytes
) -> tuple[bytes, bytes]:
    hkdf = HKDF(
        algorithm=hashes.SHA256(), length=64, salt=session_id, info=_LABEL + b"/keys" + transcript
    )
    okm = hkdf.derive(shared_secret)
    return okm[:32], okm[32:]  # client-to-server, server-to-client


class PendingSession:
    """Client side between sending HELLO and receiving ACCEPT."""

    def __init__(
        self, session: Session, transcript: bytes, server: ServerIdentity, signer: Signer
    ) -> None:
        self._session: Session | None = session
        self._transcript = transcript
        self._server = server
        self._signer = signer

    def finish(self, accept_frame: bytes) -> Session:
        """Check the server's ACCEPT frame and return the established session.

        Raises:
            TransportError: If ACCEPT is malformed, fails its AEAD tag, carries a
                signature that the pinned server key does not verify, or
                `finish` was already called.
        """
        session, self._session = self._session, None
        if session is None:
            raise TransportError("Handshake already finished")
        signature = session._open(accept_frame, MessageType.ACCEPT)
        if not self._signer.verify(
            self._server.signing_public_key, _ACCEPT_LABEL + self._transcript, signature
        ):
            raise AuthenticationError("Server signature does not verify with the pinned key")
        return session


def initiate(
    credentials: ClientCredentials,
    server: ServerIdentity,
    *,
    kem: Kem,
    signer: Signer,
    clock: Clock = time.time,
    max_clock_skew: float = DEFAULT_MAX_CLOCK_SKEW,
) -> tuple[bytes, PendingSession]:
    """Start a handshake with ``server``.

    Returns:
        The HELLO frame to send and the pending session to finish with the
        server's ACCEPT frame.

    Raises:
        ValueError: If the client id is empty or longer than
            ``MAX_CLIENT_ID_BYTES`` in UTF-8.
        InvalidKeyError: If a key has the wrong length for ``kem`` or ``signer``.
    """
    client_id = credentials.client_id.encode("utf-8")
    if not 0 < len(client_id) <= MAX_CLIENT_ID_BYTES:
        raise ValueError(f"Client id must be 1 to {MAX_CLIENT_ID_BYTES} UTF-8 bytes")
    session_id = secrets.token_bytes(SESSION_ID_SIZE)
    ciphertext, shared_secret = kem.encapsulate(server.kem_public_key)
    header = FrameHeader(VERSION, MessageType.HELLO, session_id, 0, int(clock())).pack()
    signed = header + len(client_id).to_bytes(2, "big") + client_id + ciphertext
    signature = signer.sign(credentials.signing_secret_key, signed)
    transcript = _transcript(kem, signer, server, signed)
    client_to_server, server_to_client = _session_keys(shared_secret, session_id, transcript)
    session = Session(
        session_id=session_id,
        peer_id="server",
        send_key=client_to_server,
        recv_key=server_to_client,
        clock=clock,
        max_clock_skew=max_clock_skew,
    )
    return signed + signature, PendingSession(session, transcript, server, signer)


def accept(
    hello: bytes,
    *,
    kem: Kem,
    signer: Signer,
    server: ServerIdentity,
    kem_secret_key: bytes,
    signing_secret_key: bytes,
    client_public_key: Callable[[str], bytes | None],
    clock: Clock = time.time,
    max_clock_skew: float = DEFAULT_MAX_CLOCK_SKEW,
) -> tuple[Session, bytes]:
    """Verify a client's HELLO, decapsulate, and build the ACCEPT reply.

    Args:
        hello: The HELLO frame.
        kem: The server's KEM.
        signer: The signature scheme used by clients and the server.
        server: The server's own public keys, as clients pin them.
        kem_secret_key: The server's KEM secret key.
        signing_secret_key: The server's ML-DSA identity secret key.
        client_public_key: Returns the registered ML-DSA public key for a
            client id, or None if the client is unknown.
        clock: Source of the current Unix time in seconds.
        max_clock_skew: Largest accepted difference between the HELLO
            timestamp and ``clock()``, in seconds.

    Returns:
        The server-side session (``peer_id`` is the client id) and the ACCEPT
        frame to send back.

    Raises:
        FrameError: If HELLO is malformed.
        AuthenticationError: If the client is unknown or its signature fails.
        ExpiredFrameError: If the HELLO timestamp is outside the allowed skew.
    """
    header = parse_header(hello)
    if header.msg_type != MessageType.HELLO or header.counter != 0:
        raise FrameError("Expected a HELLO frame with counter 0")
    body = hello[HEADER_SIZE:]
    if len(body) < 2:
        raise FrameError("HELLO is too short")
    id_length = int.from_bytes(body[:2], "big")
    if not 0 < id_length <= MAX_CLIENT_ID_BYTES:
        raise FrameError(f"Client id length {id_length} is out of range")
    if len(body) != 2 + id_length + kem.ciphertext_size + signer.signature_size:
        raise FrameError("HELLO length does not match the KEM and signature sizes")
    try:
        client_id = body[2 : 2 + id_length].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FrameError("Client id is not valid UTF-8") from exc
    signed_end = HEADER_SIZE + 2 + id_length + kem.ciphertext_size
    signed, signature = hello[:signed_end], hello[signed_end:]

    public_key = client_public_key(client_id)
    if public_key is None:
        raise AuthenticationError(f"Unknown client {client_id!r}")
    if not signer.verify(public_key, signed, signature):
        raise AuthenticationError(f"HELLO signature from {client_id!r} does not verify")
    _check_timestamp(header.timestamp, clock, max_clock_skew)
    try:
        shared_secret = kem.decapsulate(
            kem_secret_key, hello[signed_end - kem.ciphertext_size : signed_end]
        )
    except InvalidCiphertextError as exc:
        raise AuthenticationError(f"{client_id!r} signed an invalid KEM ciphertext") from exc

    transcript = _transcript(kem, signer, server, signed)
    client_to_server, server_to_client = _session_keys(shared_secret, header.session_id, transcript)
    session = Session(
        session_id=header.session_id,
        peer_id=client_id,
        send_key=server_to_client,
        recv_key=client_to_server,
        clock=clock,
        max_clock_skew=max_clock_skew,
    )
    reply = session._seal(
        MessageType.ACCEPT, signer.sign(signing_secret_key, _ACCEPT_LABEL + transcript)
    )
    return session, reply
