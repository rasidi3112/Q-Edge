"""Client enrollment, challenge-response login and access tokens.

An admin enrolls each client id with its ML-DSA public key (``qedge clients
add`` or ``POST /admin/clients``). To log in, a client asks for a 32-byte
nonce, signs ``label || nonce || client_id || server_id`` and exchanges the
signature for a short-lived token. Nonces live 60 s and can be used once. See
docs/decisions/0007-admin-enrollment.md.
"""

from __future__ import annotations

import logging
import secrets
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jwt
from sqlalchemy import delete
from sqlalchemy.orm import Session

from qedge.crypto.sign import Signer
from qedge.crypto.transport import MAX_CLIENT_ID_BYTES
from qedge.protocol import challenge_message
from qedge.server.db import Challenge, Client

if TYPE_CHECKING:
    from qedge.server.settings import Settings

logger = logging.getLogger(__name__)

SCOPES = ("client", "admin")
NONCE_BYTES = 32
CHALLENGE_TTL_SECONDS = 60


class TokenError(Exception):
    """A bearer token is malformed, expired, or not issued by this server."""


@dataclass(frozen=True)
class TokenClaims:
    """Verified claims of an access token."""

    subject: str
    scope: str
    token_id: str
    issued_at: int
    expires_at: int


class TokenIssuer:
    """Issues and verifies short-lived HS256 access tokens.

    Tokens carry ``iss`` (the server id), ``aud``, ``sub`` (the client id),
    ``scope``, ``jti``, ``iat`` and ``exp``; all are required on verification.

    Args:
        secret: HMAC key, at least 32 bytes (enforced by `Settings`).
        issuer: Server id written to and required in ``iss``.
        ttl_seconds: Token lifetime; also reported as ``expires_in``.
        clock: Source of the current Unix time in seconds.
    """

    AUDIENCE = "qedge-api"
    ALGORITHM = "HS256"

    def __init__(
        self, secret: str, issuer: str, ttl_seconds: int, clock: Callable[[], float] = time.time
    ) -> None:
        self._secret = secret
        self._issuer = issuer
        self.ttl_seconds = ttl_seconds
        self._clock = clock

    @classmethod
    def from_settings(cls, settings: Settings) -> TokenIssuer:
        """Build an issuer from the server settings."""
        return cls(
            settings.jwt_secret.get_secret_value(), settings.server_id, settings.jwt_ttl_seconds
        )

    def issue(self, subject: str, scope: str) -> tuple[str, int]:
        """Return ``(token, expires_in_seconds)`` for ``subject`` with ``scope``.

        Raises:
            ValueError: If ``scope`` is not one of `SCOPES`.
        """
        if scope not in SCOPES:
            raise ValueError(f"Unknown scope {scope!r}; expected {SCOPES}")
        now = int(self._clock())
        claims = {
            "iss": self._issuer,
            "aud": self.AUDIENCE,
            "sub": subject,
            "scope": scope,
            "jti": uuid.uuid4().hex,
            "iat": now,
            "exp": now + self.ttl_seconds,
        }
        return jwt.encode(claims, self._secret, algorithm=self.ALGORITHM), self.ttl_seconds

    def verify(self, token: str) -> TokenClaims:
        """Check signature, issuer, audience, expiry and scope of ``token``.

        Raises:
            TokenError: If any check fails.
        """
        try:
            claims = jwt.decode(
                token,
                self._secret,
                algorithms=[self.ALGORITHM],
                audience=self.AUDIENCE,
                issuer=self._issuer,
                # Expiry is checked below against the injected clock.
                options={
                    "require": ["iss", "aud", "sub", "jti", "iat", "exp"],
                    "verify_exp": False,
                    "verify_iat": False,
                },
                leeway=0,
            )
        except jwt.PyJWTError as exc:
            raise TokenError(str(exc)) from exc
        if claims.get("exp", 0) <= self._clock():
            raise TokenError("Token has expired")
        if claims.get("scope") not in SCOPES:
            raise TokenError("Token has no valid scope")
        return TokenClaims(
            str(claims["sub"]),
            str(claims["scope"]),
            str(claims["jti"]),
            int(claims["iat"]),
            int(claims["exp"]),
        )


class ClientConflictError(Exception):
    """A client id is already enrolled with a different key or scope."""


class AuthenticationFailed(Exception):
    """Login failed; ``reason`` is a short label for metrics, never shown to clients.

    Args:
        reason: One of ``unknown_client``, ``unknown_nonce``, ``expired_nonce``
            or ``bad_signature``.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def enroll_client(
    session: Session, client_id: str, public_key: bytes, scope: str, *, signer: Signer
) -> bool:
    """Enroll ``client_id`` with its ML-DSA public key; idempotent for the same key.

    Returns:
        True if a new client was created, False if the identical enrollment
        already existed.

    Raises:
        ValueError: If the id is empty or too long, the scope is unknown, or the
            key length does not match ``signer``.
        ClientConflictError: If the id exists with another key or scope.
    """
    if not 0 < len(client_id.encode()) <= MAX_CLIENT_ID_BYTES:
        raise ValueError(f"Client id must be 1 to {MAX_CLIENT_ID_BYTES} UTF-8 bytes")
    if scope not in SCOPES:
        raise ValueError(f"Unknown scope {scope!r}; expected {SCOPES}")
    if len(public_key) != signer.public_key_size:
        raise ValueError(f"{signer.name} public keys are {signer.public_key_size} bytes")
    existing = session.get(Client, client_id)
    if existing is not None:
        if existing.public_key == public_key and existing.scope == scope:
            return False
        raise ClientConflictError(f"{client_id!r} is already enrolled with another key or scope")
    session.add(Client(id=client_id, public_key=public_key, scope=scope, created_at=time.time()))
    session.commit()
    return True


def issue_challenge(session: Session, client_id: str, *, now: float) -> bytes:
    """Return a fresh 32-byte nonce valid for `CHALLENGE_TTL_SECONDS`.

    Unknown client ids also get a nonce, which is never stored, so the
    response does not reveal who is enrolled.
    """
    session.execute(delete(Challenge).where(Challenge.expires_at <= now))
    nonce = secrets.token_bytes(NONCE_BYTES)
    if session.get(Client, client_id) is not None:
        session.add(
            Challenge(nonce=nonce, client_id=client_id, expires_at=now + CHALLENGE_TTL_SECONDS)
        )
    session.commit()
    return nonce


def redeem_challenge(
    session: Session,
    client_id: str,
    nonce: bytes,
    signature: bytes,
    *,
    server_id: str,
    signer: Signer,
    now: float,
) -> Client:
    """Consume ``nonce`` and verify the client's signature over it.

    The nonce is deleted before the signature is checked, so each nonce can be
    tried once.

    Returns:
        The enrolled client.

    Raises:
        AuthenticationFailed: If the nonce is unknown, expired, or belongs to
            another client, or the signature does not verify.
    """
    challenge = session.get(Challenge, nonce)
    if challenge is None or challenge.client_id != client_id:
        raise AuthenticationFailed("unknown_nonce")
    deleted = session.execute(delete(Challenge).where(Challenge.nonce == nonce))
    session.commit()
    if getattr(deleted, "rowcount", 0) != 1:
        raise AuthenticationFailed("unknown_nonce")  # redeemed concurrently
    if challenge.expires_at <= now:
        raise AuthenticationFailed("expired_nonce")
    client = session.get(Client, client_id)
    if client is None:
        raise AuthenticationFailed("unknown_client")
    message = challenge_message(nonce, client_id, server_id)
    if not signer.verify(client.public_key, message, signature):
        raise AuthenticationFailed("bad_signature")
    return client
