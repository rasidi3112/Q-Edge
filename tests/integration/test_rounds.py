import json
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

from qedge.fl.remote import RemoteClient, RemoteError
from qedge.fl.weights import ModelSpec, TensorSpec, UpdateHeader, decode_weights, encode_update
from qedge.server.app import create_app
from qedge.server.db import BlobStore, Client, Database
from qedge.server.rounds import RoundNotOpenError, RoundService

SPEC = [{"name": "w", "shape": [2], "dtype": "float32"}]


@pytest.fixture
def server(settings):
    with TestClient(create_app(settings)) as client:
        yield client


@pytest.fixture
def admin(server, enroll, login):
    return login(server, "admin", enroll("admin", scope="admin"))


@pytest.fixture
def make_client(server, enroll):
    def _make(client_id):
        remote = RemoteClient(server, client_id, enroll(client_id), backend="pure")
        remote.login()
        remote.open_session()
        return remote

    return _make


def open_round(server, admin, min_clients=2, deadline_seconds=None):
    body = {"tensors": SPEC, "min_clients": min_clients, "deadline_seconds": deadline_seconds}
    response = server.post("/admin/rounds", json=body, headers=admin)
    assert response.status_code == 201, response.text
    return response.json()["round_id"]


def wait(server):
    server.app.state.qedge.rounds.wait(timeout=30)


def w(*values):
    return {"w": np.array(values, dtype=np.float32)}


def test_round_closes_at_min_clients_and_serves_the_fedavg_model(server, admin, make_client):
    round_id = open_round(server, admin, min_clients=2)
    a, b = make_client("hospital_A"), make_client("hospital_B")

    assert a.submit(round_id, w(1.0, 1.0), n_samples=1, local_loss=0.5)["state"] == "open"
    assert b.submit(round_id, w(3.0, 3.0), n_samples=3, local_loss=0.4)["state"] == "aggregating"
    wait(server)

    assert a.round(round_id)["state"] == "closed"
    np.testing.assert_allclose(a.model(round_id)["w"], [2.5, 2.5])


def test_deadline_aggregates_whatever_arrived(server, admin, make_client):
    # Log in first: the pure-Python handshake can take longer than the deadline.
    a = make_client("hospital_A")
    round_id = open_round(server, admin, min_clients=3, deadline_seconds=1.0)
    a.submit(round_id, w(4.0, 2.0), n_samples=5, local_loss=0.1)

    with server.stream("GET", f"/rounds/{round_id}/events", headers=admin) as response:
        states = [
            json.loads(line[6:])["state"]
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]
    wait(server)

    assert states[-1] == "closed"
    np.testing.assert_allclose(a.model(round_id)["w"], [4.0, 2.0])


def test_deadline_without_submissions_fails_the_round(server, admin):
    round_id = open_round(server, admin, min_clients=1, deadline_seconds=0.2)

    with server.stream("GET", f"/rounds/{round_id}/events", headers=admin) as response:
        states = [
            json.loads(line[6:])["state"]
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]

    assert states[0] == "open"
    assert states[-1] == "failed"
    assert server.get(f"/rounds/{round_id}/model", headers=admin).status_code == 409


def test_update_for_another_round_or_spec_is_rejected(server, admin, make_client):
    first = open_round(server, admin, min_clients=5)
    second = open_round(server, admin, min_clients=5)
    a = make_client("hospital_A")
    other_spec = ModelSpec((TensorSpec("w", (3,), "float32"),))

    with pytest.raises(RemoteError) as wrong_round:
        a.send_frame(second, a.seal_update(first, w(1.0, 1.0), 1, 0.1))
    with pytest.raises(RemoteError) as wrong_spec:
        a.send_frame(first, a.seal_update(first, w(1.0, 1.0, 1.0), 1, 0.1, spec=other_spec))

    assert wrong_round.value.status_code == 422
    assert wrong_spec.value.status_code == 422


def test_replayed_and_foreign_frames_are_rejected(server, admin, make_client):
    round_id = open_round(server, admin, min_clients=5)
    a, b = make_client("hospital_A"), make_client("hospital_B")
    frame = a.seal_update(round_id, w(1.0, 1.0), 1, 0.1)

    with pytest.raises(RemoteError) as foreign:
        b.send_frame(round_id, frame)
    a.send_frame(round_id, frame)
    with pytest.raises(RemoteError) as replay:
        a.send_frame(round_id, frame)

    assert foreign.value.status_code == 401
    assert replay.value.status_code == 409


def test_handshake_for_another_client_id_is_rejected(server, enroll, signer):
    victim = RemoteClient(server, "hospital_A", enroll("hospital_A"), backend="pure")
    attacker = RemoteClient(server, "hospital_B", enroll("hospital_B"), backend="pure")
    attacker.login()
    victim._headers = attacker.headers  # attacker's token, victim's signed HELLO

    with pytest.raises(RemoteError) as refused:
        victim.open_session()

    assert refused.value.status_code == 401


def test_oversized_update_is_rejected(settings, enroll):
    small = settings.model_copy(update={"max_update_bytes": 1024})
    with TestClient(create_app(small)) as server:
        a = RemoteClient(server, "hospital_A", enroll("hospital_A"), backend="pure")
        a.login()
        a.open_session()

        with pytest.raises(RemoteError) as too_large:
            a.send_frame(1, b"\0" * 2048)

    assert too_large.value.status_code == 413


def test_concurrent_submissions_are_never_lost(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'qedge.db'}")
    database.upgrade()
    service = RoundService(database, BlobStore(tmp_path / "blobs"), max_samples=100)
    spec = ModelSpec((TensorSpec("w", (2,), "float32"),))
    with database.session() as session:
        for i in range(10):
            session.add(Client(id=f"c{i}", public_key=b"k", scope="client", created_at=0.0))
        session.commit()
    round_id = service.create(spec, min_clients=5, deadline_seconds=None).round_id
    accepted, refused = [], []

    def submit(i):
        payload = encode_update(UpdateHeader(round_id, 1, 0.0), spec, w(float(i), float(i)))
        try:
            service.submit(round_id, f"c{i}", payload)
            accepted.append(i)
        except RoundNotOpenError:
            refused.append(i)

    threads = [threading.Thread(target=submit, args=(i,)) for i in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    service.wait(timeout=30)
    info = service.info(round_id)

    assert len(accepted) == 5 and len(refused) == 5
    assert info.state == "closed" and info.submissions == 5
    expected = np.mean([float(i) for i in accepted])
    np.testing.assert_allclose(
        decode_weights(spec, service.model(round_id))["w"], [expected] * 2, rtol=1e-6
    )
    service.shutdown()
    database.dispose()


@pytest.mark.parametrize(
    ("aggregation", "expected"), [("fedavg", [102.2 / 3, -98.2 / 3]), ("median", [1.2, 0.8])]
)
def test_server_aggregation_rule_is_configurable(tmp_path, aggregation, expected):
    database = Database(f"sqlite:///{tmp_path / 'qedge.db'}")
    database.upgrade()
    service = RoundService(
        database, BlobStore(tmp_path / "blobs"), max_samples=100, aggregation=aggregation
    )
    spec = ModelSpec((TensorSpec("w", (2,), "float32"),))
    with database.session() as session:
        for i in range(3):
            session.add(Client(id=f"c{i}", public_key=b"k", scope="client", created_at=0.0))
        session.commit()
    round_id = service.create(spec, min_clients=3, deadline_seconds=None).round_id
    for i, values in enumerate([(1.0, 1.0), (1.2, 0.8), (100.0, -100.0)]):
        service.submit(
            round_id, f"c{i}", encode_update(UpdateHeader(round_id, 1, 0.0), spec, w(*values))
        )
    service.wait(timeout=30)

    model = decode_weights(spec, service.model(round_id))["w"]
    np.testing.assert_allclose(model, expected, rtol=1e-3)
    service.shutdown()
    database.dispose()


def test_unknown_server_aggregation_is_rejected(tmp_path):
    from qedge.fl.robust import server_rule

    with pytest.raises(ValueError, match="norm_clip"):
        server_rule("norm_clip")
