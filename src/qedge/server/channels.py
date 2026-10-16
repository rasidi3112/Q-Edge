"""Server-side store of encrypted transport sessions.

`qedge.crypto.transport.accept` keeps no state; this store remembers every
HELLO session id until it expires, so a recorded HELLO cannot open a second
session, and binds each session to the client that opened it.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from qedge.crypto.kem import Kem
from qedge.crypto.keys import ServerKeys
from qedge.crypto.sign import Signer
from qedge.crypto.transport import AuthenticationError, ReplayError, Session, accept, parse_header


class SessionStore:
    """Sessions keyed by session id, each owned by one client.

    Args:
        kem: The server KEM.
        signer: The signature scheme for clients and the server.
        keys: The server's key pairs.
        ttl_seconds: How long a session and its HELLO id are kept.
        clock: Source of the current Unix time in seconds.
    """

    def __init__(
        self,
        *,
        kem: Kem,
        signer: Signer,
        keys: ServerKeys,
        ttl_seconds: float,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._kem = kem
        self._signer = signer
        self._keys = keys
        self._ttl = ttl_seconds
        self._clock = clock
        self._sessions: dict[bytes, tuple[Session, float]] = {}
        self._lock = threading.Lock()

    def handshake(
        self, hello: bytes, client_id: str, client_public_key: Callable[[str], bytes | None]
    ) -> bytes:
        """Accept a HELLO from ``client_id`` and return the ACCEPT frame.

        Raises:
            TransportError: If the HELLO is malformed, unauthenticated, names
                another client, or reuses a session id.
        """
        header = parse_header(hello)
        with self._lock:
            self._expire()
            if header.session_id in self._sessions:
                raise ReplayError("Session id was already used")
        session, reply = accept(
            hello,
            kem=self._kem,
            signer=self._signer,
            server=self._keys.identity(),
            kem_secret_key=self._keys.kem_secret_key,
            signing_secret_key=self._keys.signing_secret_key,
            client_public_key=client_public_key,
            clock=self._clock,
        )
        if session.peer_id != client_id:
            raise AuthenticationError("HELLO names a different client than the token")
        with self._lock:
            if header.session_id in self._sessions:
                raise ReplayError("Session id was already used")
            self._sessions[header.session_id] = (session, self._clock())
        return reply

    def open(self, frame: bytes, client_id: str) -> bytes:
        """Decrypt a DATA frame sent by ``client_id`` on one of its sessions.

        Raises:
            TransportError: If the session is unknown, expired, owned by another
                client, or the frame fails any transport check.
        """
        header = parse_header(frame)
        with self._lock:
            self._expire()
            entry = self._sessions.get(header.session_id)
            if entry is None:
                raise AuthenticationError("Unknown or expired session")
            session = entry[0]
            if session.peer_id != client_id:
                raise AuthenticationError("Session belongs to another client")
            return session.open(frame)

    def _expire(self) -> None:
        cutoff = self._clock() - self._ttl
        for session_id in [sid for sid, (_, created) in self._sessions.items() if created < cutoff]:
            del self._sessions[session_id]
