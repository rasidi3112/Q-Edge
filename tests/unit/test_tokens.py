import jwt
import pytest
from pydantic import ValidationError

from qedge.server.auth import TokenError, TokenIssuer
from qedge.server.settings import Settings

SECRET = "unit-test-secret-0123456789abcdef0123"


def issuer(now=1_000_000.0, secret=SECRET, server_id="qedge"):
    return TokenIssuer(secret, server_id, ttl_seconds=600, clock=lambda: now)


def test_token_carries_required_claims_and_matching_expiry():
    token, expires_in = issuer().issue("hospital_A", "client")
    claims = jwt.decode(
        token, SECRET, algorithms=["HS256"], audience="qedge-api", options={"verify_exp": False}
    )

    assert expires_in == 600
    assert claims["exp"] - claims["iat"] == expires_in
    assert {"iss", "aud", "sub", "jti", "iat", "exp", "scope"} <= set(claims)
    assert issuer().verify(token).subject == "hospital_A"


def test_token_ids_are_unique():
    assert issuer().issue("a", "client")[0] != issuer().issue("a", "client")[0]


@pytest.mark.parametrize(
    "verifier",
    [
        issuer(now=1_000_000.0 + 601),
        issuer(secret="another-secret-0123456789abcdef0123"),
        issuer(server_id="other-server"),
    ],
)
def test_expired_foreign_or_wrong_issuer_tokens_are_rejected(verifier):
    token, _ = issuer().issue("hospital_A", "client")

    with pytest.raises(TokenError):
        verifier.verify(token)


def test_unknown_scope_is_rejected():
    with pytest.raises(ValueError, match="scope"):
        issuer().issue("hospital_A", "root")
    forged = jwt.encode(
        {
            "iss": "qedge",
            "aud": "qedge-api",
            "sub": "a",
            "jti": "j",
            "iat": 1,
            "exp": 2_000_000,
            "scope": "root",
        },
        SECRET,
        algorithm="HS256",
    )
    with pytest.raises(TokenError, match="scope"):
        issuer().verify(forged)


def test_alg_none_tokens_are_rejected():
    token = jwt.encode(
        {
            "iss": "qedge",
            "aud": "qedge-api",
            "sub": "a",
            "jti": "j",
            "iat": 1,
            "exp": 2_000_000,
            "scope": "admin",
        },
        key=None,
        algorithm="none",
    )
    with pytest.raises(TokenError):
        issuer().verify(token)


def test_tokens_cannot_be_issued_without_a_configured_secret(monkeypatch):
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)

    with pytest.raises(ValidationError, match="JWT_SECRET_KEY"):
        Settings()
    with pytest.raises(ValidationError, match="32 bytes"):
        Settings(jwt_secret="too-short")
    monkeypatch.setenv("JWT_SECRET_KEY", SECRET)
    assert Settings().jwt_secret.get_secret_value() == SECRET
