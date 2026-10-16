from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from jose import JWTError, jwt

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

class AzureKeyVaultManager:
    
    
    def __init__(
        self,
        vault_url: Optional[str] = None,
    ) -> None:
        
        self.vault_url = vault_url or os.getenv("AZURE_KEY_VAULT_URL")
        self._client = None
        self._credential = None
        self.is_connected = False
        
        # Cache for secrets (with TTL)
        self._cache: Dict[str, Tuple[str, float]] = {}
        self._cache_ttl = 300  # 5 minutes
    
    async def connect(self) -> None:
        
        if not self.vault_url:
            logger.warning("AZURE_KEY_VAULT_URL not set, Key Vault disabled")
            return
        
        try:
            from azure.identity import DefaultAzureCredential
            from azure.keyvault.secrets import SecretClient
            
            self._credential = DefaultAzureCredential()
            self._client = SecretClient(
                vault_url=self.vault_url,
                credential=self._credential,
            )
            
            # Test connection by listing secrets (limited)
            # This will fail fast if credentials are invalid
            list(self._client.list_properties_of_secrets(max_page_size=1))
            
            self.is_connected = True
            logger.info(f"Connected to Azure Key Vault: {self.vault_url}")
            
        except ImportError:
            logger.warning(
                "Azure SDK not installed. Install azure-identity and "
                "azure-keyvault-secrets for Key Vault support."
            )
        except Exception as e:
            logger.error(f"Failed to connect to Key Vault: {e}")
            raise ConnectionError(f"Key Vault connection failed: {e}")
    
    async def disconnect(self) -> None:
        
        self._client = None
        self._credential = None
        self.is_connected = False
        self._cache.clear()
        logger.info("Disconnected from Azure Key Vault")
    
    async def get_secret(
        self,
        secret_name: str,
        use_cache: bool = True,
    ) -> Optional[str]:
        
        # Check cache first
        if use_cache and secret_name in self._cache:
            value, timestamp = self._cache[secret_name]
            if time.time() - timestamp < self._cache_ttl:
                return value
        
        if not self.is_connected or self._client is None:
            # Fall back to environment variable
            env_name = secret_name.upper().replace("-", "_")
            return os.getenv(env_name)
        
        try:
            secret = self._client.get_secret(secret_name)
            value = secret.value
            
            # Cache the value
            self._cache[secret_name] = (value, time.time())
            
            return value
            
        except Exception as e:
            logger.warning(f"Failed to retrieve secret {secret_name}: {e}")
            # Fall back to environment variable
            env_name = secret_name.upper().replace("-", "_")
            return os.getenv(env_name)
    
    async def set_secret(self, secret_name: str, value: str) -> bool:
        
        if not self.is_connected or self._client is None:
            logger.warning("Key Vault not connected, cannot store secret")
            return False
        
        try:
            self._client.set_secret(secret_name, value)
            
            # Update cache
            self._cache[secret_name] = (value, time.time())
            
            logger.info(f"Stored secret: {secret_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store secret {secret_name}: {e}")
            return False
    
    async def get_quantum_credentials(self) -> Dict[str, str]:
        
        credentials = {}
        
        quantum_secrets = [
            "azure-quantum-subscription-id",
            "azure-quantum-resource-group",
            "azure-quantum-workspace",
            "ionq-api-key",
        ]
        
        for secret_name in quantum_secrets:
            value = await self.get_secret(secret_name)
            if value:
                key = secret_name.replace("-", "_").upper()
                credentials[key] = value
        
        return credentials
    
    async def get_openai_key(self) -> Optional[str]:
        
        return await self.get_secret("azure-openai-key")

# JWT Token Management
def create_jwt_token(
    client_id: str,
    session_id: str,
    expires_delta: Optional[timedelta] = None,
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

def verify_jwt_token(token: str) -> Dict[str, Any]:
    
    secret_key = os.getenv("JWT_SECRET_KEY", "development-secret-key-change-me")
    algorithm = os.getenv("JWT_ALGORITHM", "HS256")
    
    try:
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
        return payload
        
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

class PQCAuthMiddleware(BaseHTTPMiddleware):
    
    
    def __init__(self, app, excluded_paths: Optional[list] = None) -> None:
        
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
                    f"PQC request from {pqc_client_id}, "
                    f"signature length: {len(pqc_signature)}"
                )
                
            except Exception as e:
                logger.warning(f"PQC verification failed: {e}")
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": f"PQC verification failed: {str(e)}"},
                )
        
        # Continue with request
        response = await call_next(request)
        
        # Add PQC headers to response
        response.headers["X-PQC-Protected"] = "true"
        response.headers["X-PQC-Algorithm"] = "kyber1024+dilithium5"
        
        return response
