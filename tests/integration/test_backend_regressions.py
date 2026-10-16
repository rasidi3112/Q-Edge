"""Regression tests for backend findings S3, S7-S10, F6 and F7 of the 99a5be6 audit."""

import base64

import numpy as np
import pytest
from fastapi.testclient import TestClient

from qedge.fl.remote import RemoteClient, RemoteError
from qedge.fl.weights import ModelSpec, TensorSpec
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
    for response in (client.get("/"), client.get("/rounds", headers=headers)):
        assert not any(
            name.lower().startswith(("x-pqc", "x-ratelimit")) for name in response.headers
        )


def test_client_tokens_cannot_use_admin_endpoints(client, headers):
    assert client.get("/admin/clients", headers=headers).status_code == 403
    assert client.post("/admin/clients", json={}, headers=headers).status_code == 403


PUBLIC_ROUTES = {"/", "/health", "/crypto/keys", "/auth/challenge", "/auth/token"}
DOC_ROUTES = {"/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"}


def test_unauthenticated_requests_cannot_run_circuits(client):
    for route in client.app.routes:
        path = getattr(route, "path", "")
        if path in PUBLIC_ROUTES | DOC_ROUTES:
            continue
        for method in getattr(route, "methods", ()):
            url = path.replace("{round_id}", "1").replace("{job_id}", "x")
            response = client.request(method, url)
            assert response.status_code == 401, f"{method} {path} -> {response.status_code}"
    assert client.get("/fl/metrics").status_code == 404


@AUDIT
def test_quantum_jobs_bound_the_number_of_qubits(client, headers):
    job = {"circuit_type": "vqc", "parameters": {"n_qubits": 30}, "shots": 100}

    assert client.post("/quantum/submit-job", json=job, headers=headers).status_code == 422


@pytest.fixture
def remote(client, enroll, login):
    admin = login(client, "admin", enroll("admin", scope="admin"))
    body = {"tensors": [{"name": "w", "shape": [2], "dtype": "float32"}], "min_clients": 3}
    round_id = client.post("/admin/rounds", json=body, headers=admin).json()["round_id"]
    remote = RemoteClient(client, "hospital_B", enroll("hospital_B"), backend="pure")
    remote.login()
    remote.open_session()
    return remote, round_id


def _weights(*values):
    return {"w": np.array(values, dtype=np.float32)}


def test_second_submission_from_the_same_client_is_rejected(remote):
    client, round_id = remote
    client.submit(round_id, _weights(1.0, 1.0), n_samples=10, local_loss=0.1)

    with pytest.raises(RemoteError) as second:
        client.submit(round_id, _weights(-35.0, -35.0), n_samples=10, local_loss=0.1)

    assert second.value.status_code == 409


def test_inflated_sample_counts_are_rejected(remote):
    client, round_id = remote

    with pytest.raises(RemoteError) as inflated:
        client.submit(round_id, _weights(1.0, 1.0), n_samples=10**9, local_loss=0.1)

    assert inflated.value.status_code == 422


def test_mismatched_weight_shapes_are_rejected_without_a_server_error(remote):
    client, round_id = remote
    wrong = ModelSpec((TensorSpec("w", (3,), "float32"),))
    frame = client.seal_update(round_id, _weights(1.0, 2.0, 3.0), 10, 0.1, spec=wrong)

    with pytest.raises(RemoteError) as mismatch:
        client.send_frame(round_id, frame)

    assert mismatch.value.status_code == 422


def test_clients_cannot_trigger_aggregation(client, headers):
    assert client.post("/fl/trigger-aggregation", headers=headers).status_code == 404
    body = {"tensors": [{"name": "w", "shape": [1], "dtype": "float32"}], "min_clients": 1}
    assert client.post("/admin/rounds", json=body, headers=headers).status_code == 403


@AUDIT
def test_celery_routes_and_beat_name_registered_tasks():
    from qedge.server.worker import celery_app

    names = set(celery_app.tasks)
    routed = set(celery_app.conf.task_routes)
    scheduled = {entry["task"] for entry in celery_app.conf.beat_schedule.values()}

    assert routed <= names
    assert scheduled <= names
