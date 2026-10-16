import json

import pytest

from qedge.cli import main
from qedge.crypto.backends import get_kem, get_signer
from qedge.crypto.errors import BackendUnavailableError, KeyStoreError
from qedge.crypto.keys import (
    DEFAULT_KEM,
    DEFAULT_SIGNATURE,
    MANIFEST,
    fingerprint,
    generate_server_keys,
    load_server_keys,
)


def _generate(directory, overwrite=False):
    return generate_server_keys(
        directory,
        kem=get_kem(DEFAULT_KEM, "pure"),
        signer=get_signer(DEFAULT_SIGNATURE, "pure"),
        overwrite=overwrite,
    )


def test_generated_keys_load_back_unchanged(tmp_path):
    written = _generate(tmp_path)

    loaded = load_server_keys(tmp_path, backend="pure")

    assert loaded == written
    assert loaded.kem_algorithm == "X25519-ML-KEM-768"
    assert loaded.signature_algorithm == "ML-DSA-65"


def test_secret_key_files_are_private_and_keys_stay_out_of_repr(tmp_path):
    keys = _generate(tmp_path)

    for name in ("kem.key", "signing.key"):
        assert (tmp_path / name).stat().st_mode & 0o777 == 0o600
    assert keys.kem_secret_key.hex()[:16] not in repr(keys)


def test_existing_keys_are_kept_unless_overwrite_is_requested(tmp_path):
    first = _generate(tmp_path)

    with pytest.raises(KeyStoreError, match="refusing to replace"):
        _generate(tmp_path)
    second = _generate(tmp_path, overwrite=True)

    assert second.kem_public_key != first.kem_public_key
    assert load_server_keys(tmp_path, backend="pure") == second


def test_load_refuses_group_readable_secret_keys(tmp_path):
    _generate(tmp_path)
    (tmp_path / "signing.key").chmod(0o640)

    with pytest.raises(KeyStoreError, match="chmod 600"):
        load_server_keys(tmp_path, backend="pure")


def test_load_refuses_mismatched_or_truncated_keys(tmp_path):
    _generate(tmp_path / "a")
    _generate(tmp_path / "b")
    (tmp_path / "a" / "kem.key").write_bytes((tmp_path / "b" / "kem.key").read_bytes())

    with pytest.raises(KeyStoreError, match="does not belong"):
        load_server_keys(tmp_path / "a", backend="pure")

    (tmp_path / "b" / "signing.pub").write_bytes(b"short")
    with pytest.raises(KeyStoreError, match="bytes, expected"):
        load_server_keys(tmp_path / "b", backend="pure")


def test_load_refuses_public_keys_that_do_not_match_the_manifest(tmp_path):
    _generate(tmp_path / "a")
    _generate(tmp_path / "b")
    for name in ("kem.pub", "kem.key"):
        (tmp_path / "a" / name).write_bytes((tmp_path / "b" / name).read_bytes())

    with pytest.raises(KeyStoreError, match="fingerprints"):
        load_server_keys(tmp_path / "a", backend="pure")


def test_missing_or_invalid_manifest_explains_how_to_create_keys(tmp_path):
    with pytest.raises(KeyStoreError, match="qedge keys init"):
        load_server_keys(tmp_path, backend="pure")

    _generate(tmp_path)
    manifest = json.loads((tmp_path / MANIFEST).read_text())
    manifest["kem"]["algorithm"] = "Kyber1024"
    (tmp_path / MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(KeyStoreError, match="Unsupported algorithms"):
        load_server_keys(tmp_path, backend="pure")


def test_load_with_an_unavailable_backend_raises(tmp_path, monkeypatch):
    _generate(tmp_path)
    monkeypatch.setenv("OQS_INSTALL_PATH", str(tmp_path / "no-liboqs"))
    from qedge.crypto.backends import liboqs

    monkeypatch.setattr(liboqs.ctypes.util, "find_library", lambda name: None)
    liboqs._oqs.cache_clear()
    try:
        with pytest.raises(BackendUnavailableError):
            load_server_keys(tmp_path, backend="liboqs")
    finally:
        liboqs._oqs.cache_clear()


def test_cli_init_and_show(tmp_path, capsys):
    key_dir = tmp_path / "keys"

    assert main(["keys", "init", "--dir", str(key_dir), "--backend", "pure"]) == 0
    created = capsys.readouterr().out
    assert main(["keys", "show", "--dir", str(key_dir), "--backend", "pure"]) == 0
    shown = capsys.readouterr().out
    keys = load_server_keys(key_dir, backend="pure")

    assert created == shown
    assert fingerprint(keys.kem_public_key) in shown
    assert fingerprint(keys.signing_public_key) in shown


def test_cli_init_refuses_to_overwrite_without_force(tmp_path, capsys):
    args = ["keys", "init", "--dir", str(tmp_path), "--backend", "pure"]
    assert main(args) == 0

    assert main(args) == 1
    assert "refusing to replace" in capsys.readouterr().err
    assert main([*args, "--force", "--kem", "ML-KEM-1024", "--signature", "ML-DSA-87"]) == 0
    assert load_server_keys(tmp_path, backend="pure").kem_algorithm == "ML-KEM-1024"
