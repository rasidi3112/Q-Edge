import numpy as np
import pytest
from fastapi.testclient import TestClient

from qedge.fl.remote import RemoteClient, RemoteError
from qedge.server.app import create_app


def _value(text, prefix):
    lines = [line for line in text.splitlines() if line.startswith(prefix)]
    return float(lines[0].split()[-1]) if lines else 0.0


def test_metrics_count_rounds_updates_aggregations_and_auth_failures(settings, enroll, login):
    with TestClient(create_app(settings)) as server:
        admin = login(server, "admin", enroll("admin", scope="admin"))
        body = {"tensors": [{"name": "w", "shape": [1], "dtype": "float32"}], "min_clients": 1}
        round_id = server.post("/admin/rounds", json=body, headers=admin).json()["round_id"]
        client = RemoteClient(server, "hospital_A", enroll("hospital_A"), backend="pure")
        client.login()
        client.open_session()
        client.submit(round_id, {"w": np.ones(1, np.float32)}, n_samples=1, local_loss=0.0)
        server.app.state.qedge.rounds.wait(timeout=30)
        with pytest.raises(RemoteError):
            client.submit(round_id, {"w": np.ones(1, np.float32)}, n_samples=1, local_loss=0.0)
        server.get("/rounds")
        text = server.get("/metrics").text

    assert _value(text, 'qedge_rounds_total{state="opened"}') == 1
    assert _value(text, 'qedge_rounds_total{state="closed"}') == 1
    assert _value(text, 'qedge_updates_total{reason="",status="accepted"}') == 1
    assert _value(text, 'qedge_updates_total{reason="round_not_open",status="rejected"}') == 1
    assert _value(text, "qedge_aggregation_seconds_count") == 1
    assert _value(text, "qedge_round_duration_seconds_count") == 1
    assert _value(text, 'qedge_auth_failures_total{reason="missing_token"}') == 1
