"""Access tokens and a request middleware.

`PQCAuthMiddleware` does not verify signatures. Post-quantum primitives live in
`qedge.crypto`.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jwt
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
from starlette.types import ASGIApp

if TYPE_CHECKING:
    from qedge.server.settings import Settings

logger = logging.getLogger(__name__)

SCOPES = ("client", "admin")


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


class PQCAuthMiddleware(BaseHTTPMiddleware):
    """Checks ``X-PQC-Timestamp`` and adds ``X-PQC-*`` headers to every response.

    It does not verify signatures and does not reject requests that lack the
    headers, so ``X-PQC-Protected: true`` carries no meaning.
    """

    def __init__(self, app: ASGIApp, excluded_paths: list[str] | None = None) -> None:
        super().__init__(app)
        self.excluded_paths = excluded_paths or [
            "/",
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/auth/register",
        ]

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Reject requests whose ``X-PQC-Timestamp`` is more than 300 s off; pass all others."""
        # Skip excluded paths
        if request.url.path in self.excluded_paths:
            return await call_next(request)

        # Check for PQC headers
        pqc_signature = request.headers.get("X-PQC-Signature")
        pqc_timestamp = request.headers.get("X-PQC-Timestamp")
        pqc_client_id = request.headers.get("X-PQC-Client-ID")

        # For signed requests, verify signature
        if pqc_signature and pqc_timestamp and pqc_client_id:
            try:
                # Anti-replay check
                timestamp = int(pqc_timestamp)
                current_time = int(time.time())

                if abs(current_time - timestamp) > 300:  # 5 minute window
                    return JSONResponse(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        content={"detail": "Request timestamp expired"},
                    )

                # The signature is not verified; only its length is logged.
                logger.debug(
                    f"PQC request from {pqc_client_id}, signature length: {len(pqc_signature)}"
                )

            except Exception as e:
                logger.warning(f"PQC verification failed: {e}")
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": f"PQC verification failed: {e!s}"},
                )

        # Continue with request
        response = await call_next(request)

        # Add PQC headers to response
        response.headers["X-PQC-Protected"] = "true"
        response.headers["X-PQC-Algorithm"] = "kyber1024+dilithium5"

        return response
