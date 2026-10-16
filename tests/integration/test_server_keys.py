import base64

import pytest
from fastapi.testclient import TestClient

from qedge.crypto.errors import KeyStoreError
from qedge.crypto.keys import fingerprint, load_server_keys
from qedge.server.app import create_app
from qedge.server.settings import Settings


def test_server_refuses_to_start_without_a_key_directory(settings):
    app = create_app(settings.model_copy(update={"key_dir": None}))

    with pytest.raises(KeyStoreError, match="QEDGE_KEY_DIR"), TestClient(app):
        pass


def test_server_refuses_to_start_with_an_empty_key_directory(settings, tmp_path):
    app = create_app(settings.model_copy(update={"key_dir": tmp_path}))

    with pytest.raises(KeyStoreError, match="qedge keys init"), TestClient(app):
        pass


def test_unknown_backend_is_rejected_in_settings(settings, monkeypatch):
    with pytest.raises(ValueError, match=r"crypto_backend|QEDGE_CRYPTO_BACKEND"):
        Settings(jwt_secret=settings.jwt_secret, crypto_backend="simulated")
    monkeypatch.setenv("QEDGE_CRYPTO_BACKEND", "simulated")
    with pytest.raises(ValueError, match=r"crypto_backend|QEDGE_CRYPTO_BACKEND"):
        Settings(jwt_secret=settings.jwt_secret)


def test_crypto_keys_endpoint_serves_the_loaded_public_keys(settings):
    keys = load_server_keys(settings.key_dir, backend="pure")
    with TestClient(create_app(settings)) as client:
        body = client.get("/crypto/keys").json()

    assert base64.b64decode(body["kem_public_key"]) == keys.kem_public_key
    assert base64.b64decode(body["signing_public_key"]) == keys.signing_public_key
    assert body["kem_fingerprint"] == fingerprint(keys.kem_public_key)
    assert body["kem_algorithm"] == "X25519-ML-KEM-768"
    assert body["signature_algorithm"] == "ML-DSA-65"
