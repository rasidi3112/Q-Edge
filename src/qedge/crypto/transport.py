from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import secrets
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

@dataclass
class PQCSession:
    
    
    session_id: str
    shared_secret: bytes
    server_public_key: bytes = b""
    client_signing_key: bytes = b""
    client_verify_key: bytes = b""
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    message_counter: int = 0

class PQCTransportLayer:
    
    
    # Constants
    NONCE_SIZE = 12
    TAG_SIZE = 16
    TIMESTAMP_TOLERANCE = 300  # 5 minutes
    
    def __init__(
        self,
        client_id: str,
        use_simulation: bool = True,  # True for development
    ) -> None:
        
        self.client_id = client_id
        self.use_simulation = use_simulation
        self.session: Optional[PQCSession] = None
        
        # Try to import actual crypto libraries
        self._aes_available = False
        self._oqs_available = False
        
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            self._aesgcm_class = AESGCM
            self._aes_available = True
        except ImportError:
            logger.warning("cryptography library not available, using simulation")
        
        try:
            import oqs
            self._oqs = oqs
            self._oqs_available = True
        except ImportError:
            logger.warning("liboqs not available, using simulation")
        
        # Initialize client keypair for signing
        self._init_signing_keypair()
        
        logger.info(f"Initialized PQCTransportLayer for client: {client_id}")
    
    def _init_signing_keypair(self) -> None:
        
        if self._oqs_available and not self.use_simulation:
            sig = self._oqs.Signature("Dilithium5")
            self._signing_public_key = sig.generate_keypair()
            self._signing_private_key = sig.export_secret_key()
        else:
            # Simulated keypair
            self._signing_private_key = secrets.token_bytes(64)
            self._signing_public_key = hashlib.sha256(
                self._signing_private_key
            ).digest()
    
    async def establish_session(
        self,
        server_public_key: bytes,
    ) -> PQCSession:
        
        session_id = f"{self.client_id}_{int(time.time())}_{secrets.token_hex(8)}"
        
        # Perform key encapsulation
        if self._oqs_available and not self.use_simulation:
            kem = self._oqs.KeyEncapsulation("Kyber1024")
            ciphertext, shared_secret = kem.encap_secret(server_public_key)
        else:
            # Simulated key exchange
            shared_secret = hashlib.sha256(
                server_public_key + session_id.encode()
            ).digest()
        
        self.session = PQCSession(
            session_id=session_id,
            shared_secret=shared_secret,
            server_public_key=server_public_key,
            client_signing_key=self._signing_private_key,
            client_verify_key=self._signing_public_key,
        )
        
        logger.info(f"Established PQC session: {session_id}")
        return self.session
    
    def _derive_key(
        self,
        context: bytes,
        key_length: int = 32,
    ) -> bytes:
        
        if self.session is None:
            raise RuntimeError("No session established")
        
        # Simple HKDF-like derivation
        prk = hmac.new(
            self.session.shared_secret,
            context,
            hashlib.sha256,
        ).digest()
        
        # Expand to desired length
        blocks = (key_length + 31) // 32
        output = b""
        previous = b""
        
        for i in range(blocks):
            previous = hmac.new(
                prk,
                previous + context + bytes([i + 1]),
                hashlib.sha256,
            ).digest()
            output += previous
        
        return output[:key_length]
    
    def encrypt_weights(
        self,
        weights: List[NDArray],
    ) -> bytes:
        
        # Serialize weights
        import pickle
        serialized = pickle.dumps(weights)
        
        # Add compression
        import zlib
        compressed = zlib.compress(serialized, level=6)
        
        # Encrypt
        return self._encrypt(compressed, context=b"weight_encryption")
    
    def decrypt_weights(
        self,
        ciphertext: bytes,
    ) -> List[NDArray]:
        
        import pickle
        import zlib
        
        # Decrypt
        compressed = self._decrypt(ciphertext, context=b"weight_encryption")
        
        # Decompress
        serialized = zlib.decompress(compressed)
        
        # Deserialize
        return pickle.loads(serialized)
    
    def _encrypt(
        self,
        plaintext: bytes,
        context: bytes,
    ) -> bytes:
        
        # Derive encryption key
        key = self._derive_key(context)
        
        # Generate nonce
        nonce = secrets.token_bytes(self.NONCE_SIZE)
        
        if self._aes_available and not self.use_simulation:
            # Use actual AES-GCM
            aesgcm = self._aesgcm_class(key)
            ciphertext = aesgcm.encrypt(nonce, plaintext, None)
        else:
            # Simulated encryption (XOR with key stream)
            key_stream = self._generate_keystream(key, nonce, len(plaintext))
            ciphertext_raw = bytes(a ^ b for a, b in zip(plaintext, key_stream))
            
            # Simulated tag
            tag = hmac.new(key, nonce + ciphertext_raw, hashlib.sha256).digest()[:self.TAG_SIZE]
            ciphertext = ciphertext_raw + tag
        
        return nonce + ciphertext
    
    def _decrypt(
        self,
        ciphertext: bytes,
        context: bytes,
    ) -> bytes:
        
        # Derive encryption key
        key = self._derive_key(context)
        
        # Extract nonce
        nonce = ciphertext[:self.NONCE_SIZE]
        encrypted_data = ciphertext[self.NONCE_SIZE:]
        
        if self._aes_available and not self.use_simulation:
            aesgcm = self._aesgcm_class(key)
            return aesgcm.decrypt(nonce, encrypted_data, None)
        else:
            # Simulated decryption
            ciphertext_raw = encrypted_data[:-self.TAG_SIZE]
            tag = encrypted_data[-self.TAG_SIZE:]
            
            # Verify tag
            expected_tag = hmac.new(key, nonce + ciphertext_raw, hashlib.sha256).digest()[:self.TAG_SIZE]
            if not hmac.compare_digest(tag, expected_tag):
                raise ValueError("Authentication failed")
            
            # Decrypt
            key_stream = self._generate_keystream(key, nonce, len(ciphertext_raw))
            return bytes(a ^ b for a, b in zip(ciphertext_raw, key_stream))
    
    def _generate_keystream(
        self,
        key: bytes,
        nonce: bytes,
        length: int,
    ) -> bytes:
        
        keystream = b""
        counter = 0
        
        while len(keystream) < length:
            block = hmac.new(
                key,
                nonce + counter.to_bytes(8, "big"),
                hashlib.sha256,
            ).digest()
            keystream += block
            counter += 1
        
        return keystream[:length]
    
    def sign(self, data: bytes) -> bytes:
        
        if self._oqs_available and not self.use_simulation:
            sig = self._oqs.Signature("Dilithium5", self._signing_private_key)
            return sig.sign(data)
        else:
            # Simulated signature using HMAC
            return hmac.new(
                self._signing_private_key,
                data,
                hashlib.sha512,
            ).digest()
    
    def verify(
        self,
        data: bytes,
        signature: bytes,
        public_key: bytes,
    ) -> bool:
        
        if self._oqs_available and not self.use_simulation:
            sig = self._oqs.Signature("Dilithium5")
            return sig.verify(data, signature, public_key)
        else:
            # Simulated verification (check signature length)
            return len(signature) >= 64
    
    def create_authenticated_message(
        self,
        payload: bytes,
        message_type: str = "weights",
    ) -> bytes:
        
        if self.session is None:
            raise RuntimeError("No session established")
        
        # Update session state
        self.session.message_counter += 1
        self.session.last_activity = time.time()
        
        # Build header
        timestamp = int(time.time())
        header = struct.pack(
            ">I32sI64s",
            timestamp,
            message_type.encode().ljust(32, b'\0'),
            self.session.message_counter,
            self.client_id.encode().ljust(64, b'\0'),
        )
        
        # Encrypt payload
        encrypted_payload = self._encrypt(payload, context=b"message_payload")
        
        # Sign header + encrypted payload
        signature = self.sign(header + encrypted_payload)
        
        # Combine: header || encrypted_payload || signature
        return header + encrypted_payload + signature
    
    def parse_authenticated_message(
        self,
        message: bytes,
    ) -> Tuple[bytes, Dict[str, Any]]:
        
        # Header is 104 bytes (4 + 32 + 4 + 64)
        header_size = 104
        
        if len(message) < header_size + self.NONCE_SIZE + self.TAG_SIZE:
            raise ValueError("Message too short")
        
        # Parse header
        header = message[:header_size]
        timestamp, msg_type, counter, client_id = struct.unpack(
            ">I32sI64s",
            header,
        )
        
        # Check timestamp
        current_time = int(time.time())
        if abs(current_time - timestamp) > self.TIMESTAMP_TOLERANCE:
            raise ValueError("Message timestamp expired")
        
        # Find signature (assuming fixed 64-byte signature for simulation)
        signature_size = 64
        signature = message[-signature_size:]
        encrypted_payload = message[header_size:-signature_size]
        
        # Verify signature
        if not self.verify(
            header + encrypted_payload,
            signature,
            self.session.server_public_key if self.session else b"",
        ):
            raise ValueError("Signature verification failed")
        
        # Decrypt payload
        payload = self._decrypt(encrypted_payload, context=b"message_payload")
        
        metadata = {
            "timestamp": timestamp,
            "message_type": msg_type.rstrip(b'\0').decode(),
            "counter": counter,
            "client_id": client_id.rstrip(b'\0').decode(),
        }
        
        return payload, metadata
    
    def get_session_info(self) -> Dict[str, Any]:
        
        if self.session is None:
            return {"status": "no_session"}
        
        return {
            "session_id": self.session.session_id,
            "created_at": self.session.created_at,
            "last_activity": self.session.last_activity,
            "message_count": self.session.message_counter,
            "age_seconds": time.time() - self.session.created_at,
        }
    
    def get_public_key(self) -> bytes:
        
        return self._signing_public_key
    
    def get_public_key_base64(self) -> str:
        
        return base64.b64encode(self._signing_public_key).decode("ascii")
