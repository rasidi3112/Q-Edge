from fastapi.testclient import TestClient

from qedge.server.app import create_app


def test_lifespan_runs_when_client_is_used_as_context_manager(settings):
    app = create_app(settings)
    with TestClient(app) as client:
        response = client.get("/crypto/keys")

    assert response.status_code == 200


def test_app_instances_do_not_share_enrolled_clients(settings, tmp_path, enroll, login):
    secret_key = enroll("hospital_A")
    other = settings.model_copy(update={"data_dir": tmp_path / "other"})
    with TestClient(create_app(settings)) as first, TestClient(create_app(other)) as second:
        headers = login(first, "hospital_A", secret_key)

        assert first.get("/rounds", headers=headers).status_code == 200
        assert second.get("/rounds", headers=headers).status_code == 401
