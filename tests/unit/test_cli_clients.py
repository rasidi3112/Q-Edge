from qedge.cli import main
from qedge.crypto.keys import fingerprint
from qedge.server.db import Client, Database


def test_keygen_then_add_enrolls_the_client(tmp_path, capsys):
    url = f"sqlite:///{tmp_path / 'qedge.db'}"
    assert main(["clients", "keygen", "--out", str(tmp_path / "keys"), "--backend", "pure"]) == 0
    public_key = (tmp_path / "keys" / "client.pub").read_bytes()
    assert (tmp_path / "keys" / "client.key").stat().st_mode & 0o777 == 0o600

    args = ["clients", "add", "--id", "hospital_A", "--backend", "pure", "--database-url", url]
    assert main([*args, "--public-key", str(tmp_path / "keys" / "client.pub")]) == 0
    assert main([*args, "--public-key", str(tmp_path / "keys" / "client.pub")]) == 0
    assert "already enrolled" in capsys.readouterr().out

    database = Database(url)
    with database.session() as session:
        assert session.get(Client, "hospital_A").public_key == public_key
    database.dispose()
    assert fingerprint(public_key)


def test_add_refuses_a_second_key_for_the_same_id(tmp_path, capsys):
    url = f"sqlite:///{tmp_path / 'qedge.db'}"
    for name in ("a", "b"):
        assert main(["clients", "keygen", "--out", str(tmp_path / name), "--backend", "pure"]) == 0
    args = ["clients", "add", "--id", "hospital_A", "--backend", "pure", "--database-url", url]

    assert main([*args, "--public-key", str(tmp_path / "a" / "client.pub")]) == 0
    assert main([*args, "--public-key", str(tmp_path / "b" / "client.pub")]) == 1
    assert "already enrolled with another key" in capsys.readouterr().err
