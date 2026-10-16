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


def test_circuit_job_is_queued_on_celery(client, headers):
    job = {"n_qubits": 4, "n_layers": 2, "features": [0.1, 0.2]}
    with patch("qedge.server.app.run_circuit") as task:
        task.delay.return_value.id = "job-1"
        response = client.post("/quantum/jobs", json=job, headers=headers)

    assert response.status_code == 202
    assert response.json() == {
        "job_id": "job-1",
        "status": "PENDING",
        "result": None,
        "error": None,
    }
    task.delay.assert_called_once_with(4, 2, [0.1, 0.2], None, None)
