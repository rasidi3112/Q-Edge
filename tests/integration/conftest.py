import base64

import pytest

from qedge.crypto.backends import get_kem, get_signer
from qedge.crypto.keys import DEFAULT_KEM, DEFAULT_SIGNATURE, generate_server_keys
from qedge.protocol import challenge_message
from qedge.server.auth import enroll_client
from qedge.server.db import Database
from qedge.server.settings import Settings

JWT_SECRET = "integration-test-secret-0123456789abcdef"


@pytest.fixture(scope="session")
def key_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("server-keys")
    generate_server_keys(
        directory,
        kem=get_kem(DEFAULT_KEM, "pure"),
        signer=get_signer(DEFAULT_SIGNATURE, "pure"),
    )
    return directory


@pytest.fixture
def settings(key_dir, tmp_path):
    return Settings(
        jwt_secret=JWT_SECRET,
        deadline_poll_seconds=0.05,
        events_poll_seconds=0.05,
        crypto_backend="pure",
        key_dir=key_dir,
        data_dir=tmp_path / "data",
    )


@pytest.fixture(scope="session")
def signer():
    return get_signer(DEFAULT_SIGNATURE, "pure")


@pytest.fixture
def enroll(settings, signer):
    """Enroll an identity directly in the database; returns its secret key."""

    def _enroll(client_id, scope="client"):
        public_key, secret_key = signer.generate_keypair()
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        database = Database(settings.resolved_database_url())
        database.upgrade()
        with database.session() as session:
            enroll_client(session, client_id, public_key, scope, signer=signer)
        database.dispose()
        return secret_key

    return _enroll


@pytest.fixture
def login(signer, settings):
    """Run challenge-response against ``client`` and return bearer headers."""

    def _login(client, client_id, secret_key):
        challenge = client.post("/auth/challenge", json={"client_id": client_id}).json()
        nonce = base64.b64decode(challenge["nonce"])
        signature = signer.sign(secret_key, challenge_message(nonce, client_id, settings.server_id))
        response = client.post(
            "/auth/token",
            json={
                "client_id": client_id,
                "nonce": challenge["nonce"],
                "signature": base64.b64encode(signature).decode(),
            },
        )
        assert response.status_code == 200, response.text
        return {"Authorization": f"Bearer {response.json()['access_token']}"}

    return _login
