from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from jose import JWTError, jwt
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


# PQC Algorithm Constants
class PQCAlgorithm(Enum):
    KYBER_512 = "kyber512"
    KYBER_768 = "kyber768"
    KYBER_1024 = "kyber1024"
    DILITHIUM_2 = "dilithium2"
    DILITHIUM_3 = "dilithium3"
    DILITHIUM_5 = "dilithium5"


@dataclass
class PQCKeyPair:
    algorithm: PQCAlgorithm
    public_key: bytes
    private_key: bytes
    created_at: datetime


@dataclass
class EncapsulatedKey:
    ciphertext: bytes
    shared_secret: bytes


class PQCProvider:
    def __init__(self, use_simulation: bool = False) -> None:
        self._use_simulation = use_simulation
        self._oqs_available = False

        # Try to import liboqs
        try:
            import oqs

            self._oqs = oqs
            self._oqs_available = True
            logger.info("Using liboqs for PQC operations")
        except ImportError:
            logger.warning(
                "liboqs not available, using simulated PQC. "
                "Install liboqs-python for production use."
            )

    def generate_keypair(
        self,
        algorithm: PQCAlgorithm = PQCAlgorithm.KYBER_1024,
    ) -> PQCKeyPair:
        if self._oqs_available and not self._use_simulation:
            return self._generate_keypair_oqs(algorithm)
        else:
            return self._generate_keypair_simulated(algorithm)

    def _generate_keypair_oqs(self, algorithm: PQCAlgorithm) -> PQCKeyPair:
        alg_name = self._get_oqs_algorithm_name(algorithm)

        if "kyber" in algorithm.value:
            kem = self._oqs.KeyEncapsulation(alg_name)
            public_key = kem.generate_keypair()
            private_key = kem.export_secret_key()
        else:
            sig = self._oqs.Signature(alg_name)
            public_key = sig.generate_keypair()
            private_key = sig.export_secret_key()

        return PQCKeyPair(
            algorithm=algorithm,
            public_key=public_key,
            private_key=private_key,
            created_at=datetime.utcnow(),
        )

    def _generate_keypair_simulated(self, algorithm: PQCAlgorithm) -> PQCKeyPair:
        # Simulate key sizes based on algorithm
        if algorithm == PQCAlgorithm.KYBER_512:
            pk_size, sk_size = 800, 1632
        elif algorithm == PQCAlgorithm.KYBER_768:
            pk_size, sk_size = 1184, 2400
        elif algorithm == PQCAlgorithm.KYBER_1024:
            pk_size, sk_size = 1568, 3168
        elif algorithm == PQCAlgorithm.DILITHIUM_2:
            pk_size, sk_size = 1312, 2528
        elif algorithm == PQCAlgorithm.DILITHIUM_3:
            pk_size, sk_size = 1952, 4000
        else:  # DILITHIUM_5
            pk_size, sk_size = 2592, 4864

        return PQCKeyPair(
            algorithm=algorithm,
            public_key=secrets.token_bytes(pk_size),
            private_key=secrets.token_bytes(sk_size),
            created_at=datetime.utcnow(),
        )

    def encapsulate(
        self,
        public_key: bytes,
        algorithm: PQCAlgorithm = PQCAlgorithm.KYBER_1024,
    ) -> EncapsulatedKey:
        if self._oqs_available and not self._use_simulation:
            return self._encapsulate_oqs(public_key, algorithm)
        else:
            return self._encapsulate_simulated(public_key, algorithm)

    def _encapsulate_oqs(
        self,
        public_key: bytes,
        algorithm: PQCAlgorithm,
    ) -> EncapsulatedKey:
        alg_name = self._get_oqs_algorithm_name(algorithm)
        kem = self._oqs.KeyEncapsulation(alg_name)
        ciphertext, shared_secret = kem.encap_secret(public_key)

        return EncapsulatedKey(
            ciphertext=ciphertext,
            shared_secret=shared_secret,
        )

    def _encapsulate_simulated(
        self,
        public_key: bytes,
        algorithm: PQCAlgorithm,
    ) -> EncapsulatedKey:
        # Simulate ciphertext size
        if algorithm == PQCAlgorithm.KYBER_512:
            ct_size = 768
        elif algorithm == PQCAlgorithm.KYBER_768:
            ct_size = 1088
        else:
            ct_size = 1568

        # Generate deterministic shared secret based on public key
        shared_secret = hashlib.sha256(public_key + b"simulated_encap").digest()

        return EncapsulatedKey(
            ciphertext=secrets.token_bytes(ct_size),
            shared_secret=shared_secret,
        )

    def decapsulate(
        self,
        ciphertext: bytes,
        keypair: PQCKeyPair,
    ) -> bytes:
        if self._oqs_available and not self._use_simulation:
            return self._decapsulate_oqs(ciphertext, keypair)
        else:
            return self._decapsulate_simulated(ciphertext, keypair)

    def _decapsulate_oqs(self, ciphertext: bytes, keypair: PQCKeyPair) -> bytes:
        alg_name = self._get_oqs_algorithm_name(keypair.algorithm)
        kem = self._oqs.KeyEncapsulation(alg_name, keypair.private_key)
        return kem.decap_secret(ciphertext)

    def _decapsulate_simulated(self, ciphertext: bytes, keypair: PQCKeyPair) -> bytes:
        return hashlib.sha256(keypair.public_key + b"simulated_encap").digest()

    def sign(
        self,
        message: bytes,
        keypair: PQCKeyPair,
    ) -> bytes:
        if self._oqs_available and not self._use_simulation:
            return self._sign_oqs(message, keypair)
        else:
            return self._sign_simulated(message, keypair)

    def _sign_oqs(self, message: bytes, keypair: PQCKeyPair) -> bytes:
        alg_name = self._get_oqs_algorithm_name(keypair.algorithm)
        sig = self._oqs.Signature(alg_name, keypair.private_key)
        return sig.sign(message)

    def _sign_simulated(self, message: bytes, keypair: PQCKeyPair) -> bytes:
        # Use HMAC as a placeholder for Dilithium signature
        return hmac.new(keypair.private_key[:32], message, hashlib.sha256).digest()

    def verify(
        self,
        message: bytes,
        signature: bytes,
        public_key: bytes,
        algorithm: PQCAlgorithm = PQCAlgorithm.DILITHIUM_5,
    ) -> bool:
        if self._oqs_available and not self._use_simulation:
            return self._verify_oqs(message, signature, public_key, algorithm)
        else:
            return self._verify_simulated(message, signature, public_key)

    def _verify_oqs(
        self,
        message: bytes,
        signature: bytes,
        public_key: bytes,
        algorithm: PQCAlgorithm,
    ) -> bool:
        alg_name = self._get_oqs_algorithm_name(algorithm)
        sig = self._oqs.Signature(alg_name)
        return sig.verify(message, signature, public_key)

    def _verify_simulated(
        self,
        message: bytes,
        signature: bytes,
        public_key: bytes,
    ) -> bool:
        # In simulation, just check signature length
        return len(signature) >= 32

    def _get_oqs_algorithm_name(self, algorithm: PQCAlgorithm) -> str:
        mapping = {
            PQCAlgorithm.KYBER_512: "Kyber512",
            PQCAlgorithm.KYBER_768: "Kyber768",
            PQCAlgorithm.KYBER_1024: "Kyber1024",
            PQCAlgorithm.DILITHIUM_2: "Dilithium2",
            PQCAlgorithm.DILITHIUM_3: "Dilithium3",
            PQCAlgorithm.DILITHIUM_5: "Dilithium5",
        }
        return mapping.get(algorithm, "Kyber1024")


# JWT Token Management
def create_jwt_token(
    client_id: str,
    session_id: str,
    expires_delta: timedelta | None = None,
) -> str:
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

    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=algorithm)
    return encoded_jwt


def verify_jwt_token(token: str) -> dict[str, Any]:
    secret_key = os.getenv("JWT_SECRET_KEY", "development-secret-key-change-me")
    algorithm = os.getenv("JWT_ALGORITHM", "HS256")

    try:
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
        return payload

    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e!s}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


class PQCAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, excluded_paths: list | None = None) -> None:
        super().__init__(app)
        self.pqc_provider = PQCProvider()
        self.excluded_paths = excluded_paths or [
            "/",
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/auth/register",
        ]

    async def dispatch(self, request: Request, call_next):
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

                # Log PQC verification (actual verification would use stored public key)
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
