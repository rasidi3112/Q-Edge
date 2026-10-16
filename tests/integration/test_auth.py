import base64

import pytest
from fastapi.testclient import TestClient

from qedge.protocol import challenge_message
from qedge.server.app import create_app
from qedge.server.auth import AuthenticationFailed, issue_challenge, redeem_challenge
from qedge.server.db import Database


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def _answer(client, signer, secret_key, client_id, server_id="qedge"):
    challenge = client.post("/auth/challenge", json={"client_id": client_id}).json()
    nonce = base64.b64decode(challenge["nonce"])
    signature = signer.sign(secret_key, challenge_message(nonce, client_id, server_id))
    return {
        "client_id": client_id,
        "nonce": challenge["nonce"],
        "signature": base64.b64encode(signature).decode(),
    }


def test_login_issues_a_token_with_the_enrolled_scope(client, enroll, signer):
    secret_key = enroll("hospital_A")

    response = client.post("/auth/token", json=_answer(client, signer, secret_key, "hospital_A"))

    assert response.status_code == 200
    assert response.json()["scope"] == "client"
    assert response.json()["expires_in"] == 900


def test_nonce_is_single_use(client, enroll, signer):
    answer = _answer(client, signer, enroll("hospital_A"), "hospital_A")

    assert client.post("/auth/token", json=answer).status_code == 200
    assert client.post("/auth/token", json=answer).status_code == 401


@pytest.mark.parametrize("server_id", ["other-server"])
def test_signature_bound_to_another_server_is_rejected(client, enroll, signer, server_id):
    answer = _answer(client, signer, enroll("hospital_A"), "hospital_A", server_id=server_id)

    assert client.post("/auth/token", json=answer).status_code == 401


def test_signature_from_another_key_is_rejected(client, enroll, signer):
    enroll("hospital_A")
    _, attacker_key = signer.generate_keypair()

    answer = _answer(client, signer, attacker_key, "hospital_A")

    assert client.post("/auth/token", json=answer).status_code == 401


def test_unknown_client_gets_a_nonce_but_no_token(client, signer):
    _, secret_key = signer.generate_keypair()
    answer = _answer(client, signer, secret_key, "nobody")

    assert client.post("/auth/token", json=answer).status_code == 401


def test_expired_nonce_is_rejected(settings, enroll, signer):
    secret_key = enroll("hospital_A")
    database = Database(settings.resolved_database_url())
    with database.session() as session:
        nonce = issue_challenge(session, "hospital_A", now=1000.0)
        signature = signer.sign(secret_key, challenge_message(nonce, "hospital_A", "qedge"))
        with pytest.raises(AuthenticationFailed, match="expired"):
            redeem_challenge(
                session,
                "hospital_A",
                nonce,
                signature,
                server_id="qedge",
                signer=signer,
                now=1061.0,
            )
    database.dispose()


def test_requests_without_or_with_bad_tokens_are_401(client):
    assert client.get("/admin/clients").status_code == 401
    assert (
        client.get("/admin/clients", headers={"Authorization": "Bearer x.y.z"}).status_code == 401
    )


def test_admin_enrollment_is_idempotent_and_rejects_new_keys(client, enroll, login, signer):
    admin = login(client, "admin", enroll("admin", scope="admin"))
    public_key, _ = signer.generate_keypair()
    other_key, _ = signer.generate_keypair()
    body = {"client_id": "hospital_B", "public_key": base64.b64encode(public_key).decode()}

    assert client.post("/admin/clients", json=body, headers=admin).status_code == 201
    assert client.post("/admin/clients", json=body, headers=admin).status_code == 200
    conflict = {**body, "public_key": base64.b64encode(other_key).decode()}
    assert client.post("/admin/clients", json=conflict, headers=admin).status_code == 409
    short = {**body, "client_id": "hospital_C", "public_key": base64.b64encode(b"x").decode()}
    assert client.post("/admin/clients", json=short, headers=admin).status_code == 422
    listed = client.get("/admin/clients", headers=admin).json()
    assert [entry["client_id"] for entry in listed] == ["admin", "hospital_B"]
