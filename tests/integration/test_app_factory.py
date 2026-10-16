import pytest
from fastapi.testclient import TestClient

REGISTRATION = {
    "client_id": "hospital_A",
    "device_info": {},
    "public_key": "KYBER_KEY...",
    "signature": "DILITHIUM_SIG...",
}


@pytest.mark.xfail(strict=True, reason="I5: no create_app factory; state lives in module globals")
def test_lifespan_runs_when_client_is_used_as_context_manager():
    from qedge.server.app import create_app
    from qedge.server.settings import Settings

    app = create_app(Settings(n_qubits=4, vqc_layers=2))
    with TestClient(app) as client:
        response = client.get("/fl/metrics")

    assert response.status_code == 200


@pytest.mark.xfail(strict=True, reason="I5: no create_app factory; state lives in module globals")
def test_app_instances_do_not_share_registered_clients():
    from qedge.server.app import create_app
    from qedge.server.settings import Settings

    settings = Settings(n_qubits=4, vqc_layers=2)
    with TestClient(create_app(settings)) as first, TestClient(create_app(settings)) as second:
        token = first.post("/auth/register", json=REGISTRATION).json()["access_token"]
        response = second.get("/fl/global-model", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 401
