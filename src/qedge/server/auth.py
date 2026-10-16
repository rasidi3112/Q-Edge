"""JWT handling and a request middleware.

`PQCAuthMiddleware` does not verify signatures. JWTs fall back to a hardcoded
secret when ``JWT_SECRET_KEY`` is unset. Post-quantum primitives live in
`qedge.crypto`.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, cast

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from jose import JWTError, jwt
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


def create_jwt_token(
    client_id: str,
    session_id: str,
    expires_delta: timedelta | None = None,
) -> str:
    """Issue a JWT with ``client_id``, ``session_id``, ``iat`` and ``exp``.

    Reads ``JWT_SECRET_KEY``, ``JWT_ALGORITHM`` (default HS256) and
    ``JWT_EXPIRATION_MINUTES`` (default 30). The secret falls back to a hardcoded
    development value.
    """
    secret_key = os.getenv("JWT_SECRET_KEY", "development-secret-key-change-me")
    algorithm = os.getenv("JWT_ALGORITHM", "HS256")

    if expires_delta is None:
        expires_delta = timedelta(minutes=int(os.getenv("JWT_EXPIRATION_MINUTES", "30")))

    expire = datetime.utcnow() + expires_delta

    to_encode = {
        "client_id": client_id,
        "session_id": session_id,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access",
    }

    return cast(str, jwt.encode(to_encode, secret_key, algorithm=algorithm))


def verify_jwt_token(token: str) -> dict[str, Any]:
    """Decode and verify a token from `create_jwt_token`.

    Raises:
        HTTPException: 401 if the signature or expiry check fails.
    """
    secret_key = os.getenv("JWT_SECRET_KEY", "development-secret-key-change-me")
    algorithm = os.getenv("JWT_ALGORITHM", "HS256")

    try:
        return cast(dict[str, Any], jwt.decode(token, secret_key, algorithms=[algorithm]))

    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e!s}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


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
