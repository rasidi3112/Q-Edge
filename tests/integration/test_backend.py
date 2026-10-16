

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
import numpy as np

# Import the app
from qedge.server.app import app

@pytest.fixture
def client():
    
    return TestClient(app)

@pytest.fixture
def auth_headers():
    
    from qedge.server.auth import create_jwt_token
    
    token = create_jwt_token(
        client_id="test_client",
        session_id="test_session",
    )
    
    return {"Authorization": f"Bearer {token}"}

class TestHealthEndpoints:
    
    
    def test_root_endpoint(self, client):
        
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Q-Edge API"
        assert "version" in data
    
    def test_health_check(self, client):
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "services" in data
        assert "timestamp" in data

class TestAuthenticationEndpoints:
    
    
    def test_client_registration(self, client):
        
        registration_data = {
            "client_id": "mobile_001",
            "device_info": {"os": "android", "version": "13"},
            "public_key": "KYBER_PUBLIC_KEY_BASE64_ENCODED...",
            "signature": "DILITHIUM_SIGNATURE_BASE64_ENCODED...",
        }
        
        response = client.post("/auth/register", json=registration_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "session_id" in data
        assert data["token_type"] == "bearer"
    
    def test_registration_without_signature_fails(self, client):
        
        registration_data = {
            "client_id": "mobile_001",
            "device_info": {},
            "public_key": "KYBER_PUBLIC_KEY...",
            "signature": "",  # Empty signature
        }
        
        response = client.post("/auth/register", json=registration_data)
        
        assert response.status_code == 401

class TestFederatedLearningEndpoints:
    
    
    def test_submit_weights_requires_auth(self, client):
        
        submission = {
            "client_id": "test_client",
            "round_number": 1,
            "weights": [0.1, 0.2, 0.3],
            "n_samples": 100,
            "local_loss": 0.5,
        }
        
        response = client.post("/fl/submit-weights", json=submission)
        
        # Should fail without auth
        assert response.status_code in [401, 403]
    
    def test_submit_weights_with_auth(self, client, auth_headers):
        
        # First register the client
        registration_data = {
            "client_id": "test_client",
            "device_info": {},
            "public_key": "KYBER_KEY...",
            "signature": "DILITHIUM_SIG...",
        }
        client.post("/auth/register", json=registration_data)
        
        submission = {
            "client_id": "test_client",
            "round_number": 1,
            "weights": [0.1, 0.2, 0.3],
            "n_samples": 100,
            "local_loss": 0.5,
        }
        
        response = client.post(
            "/fl/submit-weights",
            json=submission,
            headers=auth_headers,
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "accepted"
    
    def test_get_global_model_not_found(self, client, auth_headers):
        
        # Register client first
        registration_data = {
            "client_id": "test_client",
            "device_info": {},
            "public_key": "KYBER_KEY...",
            "signature": "DILITHIUM_SIG...",
        }
        client.post("/auth/register", json=registration_data)
        
        response = client.get("/fl/global-model", headers=auth_headers)
        
        # Should return 404 when no model exists
        assert response.status_code == 404
    
    def test_fl_metrics(self, client):
        
        response = client.get("/fl/metrics")
        
        assert response.status_code == 200

class TestQuantumEndpoints:
    
    
    def test_submit_quantum_job_requires_auth(self, client):
        
        job_data = {
            "circuit_type": "vqc",
            "parameters": {"n_qubits": 4},
            "target_hardware": "simulator",
            "shots": 1024,
        }
        
        response = client.post("/quantum/submit-job", json=job_data)
        
        assert response.status_code in [401, 403]
    
    def test_submit_quantum_job(self, client, auth_headers):
        
        # Register client
        registration_data = {
            "client_id": "test_client",
            "device_info": {},
            "public_key": "KYBER_KEY...",
            "signature": "DILITHIUM_SIG...",
        }
        client.post("/auth/register", json=registration_data)
        
        job_data = {
            "circuit_type": "vqc",
            "parameters": {"n_qubits": 4},
            "target_hardware": "simulator",
            "shots": 1024,
        }
        
        with patch("qedge.server.app.submit_quantum_aggregation") as mock_task:
            mock_task.delay.return_value.id = "test-job-id"
            
            response = client.post(
                "/quantum/submit-job",
                json=job_data,
                headers=auth_headers,
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"

class TestSecurityMiddleware:
    
    
    def test_rate_limit_headers(self, client):
        
        response = client.get("/")
        
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
    
    def test_timing_header(self, client):
        
        response = client.get("/")
        
        assert "X-Process-Time" in response.headers
    
    def test_pqc_protected_header(self, client, auth_headers):
        
        # Register client
        registration_data = {
            "client_id": "test_client",
            "device_info": {},
            "public_key": "KYBER_KEY...",
            "signature": "DILITHIUM_SIG...",
        }
        client.post("/auth/register", json=registration_data)
        
        response = client.get("/fl/metrics", headers=auth_headers)
        
        # PQC headers should be present
        assert "X-PQC-Protected" in response.headers

class TestSecurity:
    
    
    def test_jwt_token_creation(self):
        
        from qedge.server.auth import create_jwt_token
        
        token = create_jwt_token(
            client_id="test",
            session_id="session123",
        )
        
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_jwt_token_verification(self):
        
        from qedge.server.auth import create_jwt_token, verify_jwt_token
        
        token = create_jwt_token(
            client_id="test_client",
            session_id="session123",
        )
        
        payload = verify_jwt_token(token)
        
        assert payload["client_id"] == "test_client"
        assert payload["session_id"] == "session123"
    
    def test_pqc_provider_keypair_generation(self):
        
        from qedge.server.auth import PQCProvider, PQCAlgorithm
        
        provider = PQCProvider(use_simulation=True)
        keypair = provider.generate_keypair(PQCAlgorithm.KYBER_1024)
        
        assert keypair.public_key is not None
        assert keypair.private_key is not None
        assert len(keypair.public_key) > 0
    
    def test_pqc_encapsulation(self):
        
        from qedge.server.auth import PQCProvider, PQCAlgorithm
        
        provider = PQCProvider(use_simulation=True)
        keypair = provider.generate_keypair(PQCAlgorithm.KYBER_1024)
        
        encapsulated = provider.encapsulate(
            keypair.public_key,
            PQCAlgorithm.KYBER_1024,
        )
        
        assert encapsulated.ciphertext is not None
        assert encapsulated.shared_secret is not None
        assert len(encapsulated.shared_secret) == 32
    
    def test_pqc_signing(self):
        
        from qedge.server.auth import PQCProvider, PQCAlgorithm
        
        provider = PQCProvider(use_simulation=True)
        keypair = provider.generate_keypair(PQCAlgorithm.DILITHIUM_5)
        
        message = b"Test message"
        signature = provider.sign(message, keypair)
        
        assert signature is not None
        assert len(signature) > 0
        
        # Verify signature
        is_valid = provider.verify(
            message,
            signature,
            keypair.public_key,
            PQCAlgorithm.DILITHIUM_5,
        )
        
        assert is_valid

# Run tests with: pytest tests/integration/test_backend.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
