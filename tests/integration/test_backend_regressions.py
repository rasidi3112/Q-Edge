"""Regression tests for backend findings S3, S7-S10, F6 and F7 of the 99a5be6 audit."""

import pytest
from fastapi.testclient import TestClient

from qedge.server.app import create_app

AUDIT = pytest.mark.xfail(strict=True, reason="backend audit finding")


def _register(client, client_id="hospital_A", public_key="KEY_A", signature="x"):
    body = {
        "client_id": client_id,
        "device_info": {},
        "public_key": public_key,
        "signature": signature,
    }
    return client.post("/auth/register", json=body)


def _auth(response):
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings), raise_server_exceptions=False) as test_client:
        yield test_client


@AUDIT
def test_register_rejects_existing_client_id_with_new_key(client):
    assert _register(client, public_key="KEY_A").status_code == 200

    assert _register(client, public_key="ATTACKER_KEY").status_code == 409


@AUDIT
def test_tokens_cannot_be_issued_without_a_configured_secret(monkeypatch):
    from qedge.server.auth import create_jwt_token

    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)

    with pytest.raises((RuntimeError, ValueError)):
        create_jwt_token(client_id="hospital_A", session_id="s")


@AUDIT
def test_responses_carry_no_fake_security_headers(client):
    response = client.get("/")

    assert "X-PQC-Protected" not in response.headers
    assert "X-RateLimit-Remaining" not in response.headers


@AUDIT
def test_client_tokens_cannot_use_admin_endpoints(client):
    headers = _auth(_register(client))

    assert client.get("/admin/clients", headers=headers).status_code == 403


@AUDIT
def test_unauthenticated_requests_cannot_run_circuits(client):
    assert client.get("/fl/metrics").status_code == 401


@AUDIT
def test_quantum_jobs_bound_the_number_of_qubits(client):
    headers = _auth(_register(client))
    job = {"circuit_type": "vqc", "parameters": {"n_qubits": 30}, "shots": 100}

    assert client.post("/quantum/submit-job", json=job, headers=headers).status_code == 422


def _submit(client, headers, weights, n_samples=10, round_number=1):
    body = {
        "client_id": "hospital_A",
        "round_number": round_number,
        "weights": weights,
        "n_samples": n_samples,
        "local_loss": 0.1,
    }
    return client.post("/fl/submit-weights", json=body, headers=headers)


@AUDIT
def test_second_submission_from_the_same_client_is_rejected(client):
    headers = _auth(_register(client))
    assert _submit(client, headers, [1.0, 1.0]).status_code == 200

    assert _submit(client, headers, [-35.0, -35.0]).status_code == 409


@AUDIT
def test_inflated_sample_counts_are_rejected(client):
    headers = _auth(_register(client))

    assert _submit(client, headers, [1.0, 1.0], n_samples=10**9).status_code in (400, 422)


@AUDIT
def test_mismatched_weight_shapes_are_rejected_without_a_server_error(client):
    headers = _auth(_register(client))
    _submit(client, headers, [1.0, 1.0])
    headers_b = _auth(_register(client, client_id="hospital_B", public_key="KEY_B"))
    body = {"client_id": "hospital_B", "round_number": 1, "weights": [1.0, 2.0, 3.0]}
    client.post(
        "/fl/submit-weights", json={**body, "n_samples": 10, "local_loss": 0.1}, headers=headers_b
    )

    response = client.post("/fl/trigger-aggregation", headers=headers)

    assert response.status_code < 500


@AUDIT
def test_clients_cannot_trigger_aggregation(client):
    headers = _auth(_register(client))
    _submit(client, headers, [1.0, 1.0])

    assert client.post("/fl/trigger-aggregation", headers=headers).status_code in (403, 404, 405)


@AUDIT
def test_celery_routes_and_beat_name_registered_tasks():
    from qedge.server.worker import celery_app

    names = set(celery_app.tasks)
    routed = set(celery_app.conf.task_routes)
    scheduled = {entry["task"] for entry in celery_app.conf.beat_schedule.values()}

    assert routed <= names
    assert scheduled <= names
