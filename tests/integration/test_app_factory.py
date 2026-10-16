from fastapi.testclient import TestClient

from qedge.server.app import create_app

REGISTRATION = {
    "client_id": "hospital_A",
    "device_info": {},
    "public_key": "KYBER_KEY...",
    "signature": "DILITHIUM_SIG...",
}


def test_lifespan_runs_when_client_is_used_as_context_manager(settings):
    app = create_app(settings)
    with TestClient(app) as client:
        response = client.get("/fl/metrics")

    assert response.status_code == 200


def test_app_instances_do_not_share_registered_clients(settings):
    with TestClient(create_app(settings)) as first, TestClient(create_app(settings)) as second:
        token = first.post("/auth/register", json=REGISTRATION).json()["access_token"]
        response = second.get("/fl/global-model", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 401
