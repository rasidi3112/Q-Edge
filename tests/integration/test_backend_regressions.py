"""Regression tests for backend findings S3, S7-S10, F6 and F7 of the 99a5be6 audit."""

import base64

import pytest
from fastapi.testclient import TestClient

from qedge.server.app import create_app

AUDIT = pytest.mark.xfail(strict=True, reason="backend audit finding")


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings), raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def headers(client, enroll, login):
    return login(client, "hospital_A", enroll("hospital_A"))


def test_register_rejects_existing_client_id_with_new_key(client, enroll, login, signer):
    admin = login(client, "admin", enroll("admin", scope="admin"))
    enroll("hospital_A")
    attacker_key, _ = signer.generate_keypair()
    body = {"client_id": "hospital_A", "public_key": base64.b64encode(attacker_key).decode()}

    assert client.post("/admin/clients", json=body, headers=admin).status_code == 409
    assert client.post("/auth/register", json=body).status_code == 404


def test_responses_carry_no_fake_security_headers(client, headers):
    for response in (client.get("/"), client.get("/fl/global-model", headers=headers)):
        assert not any(
            name.lower().startswith(("x-pqc", "x-ratelimit")) for name in response.headers
        )


def test_client_tokens_cannot_use_admin_endpoints(client, headers):
    assert client.get("/admin/clients", headers=headers).status_code == 403
    assert client.post("/admin/clients", json={}, headers=headers).status_code == 403


@AUDIT
def test_unauthenticated_requests_cannot_run_circuits(client):
    assert client.get("/fl/metrics").status_code == 401


@AUDIT
def test_quantum_jobs_bound_the_number_of_qubits(client, headers):
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
def test_second_submission_from_the_same_client_is_rejected(client, headers):
    assert _submit(client, headers, [1.0, 1.0]).status_code == 200

    assert _submit(client, headers, [-35.0, -35.0]).status_code == 409


@AUDIT
def test_inflated_sample_counts_are_rejected(client, headers):

    assert _submit(client, headers, [1.0, 1.0], n_samples=10**9).status_code in (400, 422)


@AUDIT
def test_mismatched_weight_shapes_are_rejected_without_a_server_error(
    client, headers, enroll, login
):
    _submit(client, headers, [1.0, 1.0])
    headers_b = login(client, "hospital_B", enroll("hospital_B"))
    body = {"client_id": "hospital_B", "round_number": 1, "weights": [1.0, 2.0, 3.0]}
    client.post(
        "/fl/submit-weights", json={**body, "n_samples": 10, "local_loss": 0.1}, headers=headers_b
    )

    response = client.post("/fl/trigger-aggregation", headers=headers)

    assert response.status_code < 500


@AUDIT
def test_clients_cannot_trigger_aggregation(client, headers):
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
