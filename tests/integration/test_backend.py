from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from qedge.server.app import create_app


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def headers(client, enroll, login):
    return login(client, "hospital_A", enroll("hospital_A"))


def test_root_and_health(client):
    assert client.get("/").json()["name"] == "Q-Edge API"
    health = client.get("/health").json()
    assert health["status"] in ("healthy", "degraded")


def test_timing_header_is_present(client):
    assert "X-Process-Time" in client.get("/").headers


def test_submit_quantum_job(client, headers):
    job = {"circuit_type": "vqc", "parameters": {"n_qubits": 4}, "shots": 1024}
    with patch("qedge.server.app.submit_quantum_aggregation") as task:
        task.delay.return_value.id = "job-1"
        response = client.post("/quantum/submit-job", json=job, headers=headers)

    assert response.status_code == 200
    assert response.json()["job_id"] == "job-1"
