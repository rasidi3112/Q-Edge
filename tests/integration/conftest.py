import pytest

from qedge.crypto.backends import get_kem, get_signer
from qedge.crypto.keys import DEFAULT_KEM, DEFAULT_SIGNATURE, generate_server_keys
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
def settings(key_dir):
    return Settings(
        jwt_secret=JWT_SECRET, n_qubits=4, vqc_layers=2, crypto_backend="pure", key_dir=key_dir
    )
