"""Server configuration read from environment variables (pydantic-settings)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MIN_JWT_SECRET_BYTES = 32


def _env(name: str) -> AliasChoices:
    return AliasChoices(name)


class Settings(BaseSettings):
    """Values the server reads at startup.

    Every field reads the environment variable named in its alias; keyword
    arguments use the field name. ``JWT_SECRET_KEY`` has no default.

    Raises:
        pydantic.ValidationError: If ``JWT_SECRET_KEY`` is missing or shorter
            than 32 bytes, or a value has the wrong type or range.
    """

    model_config = SettingsConfigDict(frozen=True, populate_by_name=True, extra="ignore")

    jwt_secret: SecretStr = Field(validation_alias=_env("JWT_SECRET_KEY"))
    jwt_ttl_seconds: int = Field(900, ge=60, le=86_400, validation_alias=_env("JWT_TTL_SECONDS"))
    server_id: str = Field("qedge", min_length=1, validation_alias=_env("QEDGE_SERVER_ID"))
    n_qubits: int = Field(8, ge=2, le=12, validation_alias=_env("VQC_NUM_QUBITS"))
    vqc_layers: int = Field(4, ge=1, le=6, validation_alias=_env("VQC_NUM_LAYERS"))
    fl_min_clients: int = Field(1, ge=1, validation_alias=_env("FL_MIN_CLIENTS"))
    cors_origins: tuple[str, ...] = Field(
        ("http://localhost:3000",), validation_alias=_env("CORS_ORIGINS")
    )
    rate_limit_requests: int = Field(100, validation_alias=_env("RATE_LIMIT_REQUESTS"))
    debug: bool = Field(False, validation_alias=_env("DEBUG_MODE"))
    crypto_backend: Literal["liboqs", "pure"] = Field(
        "liboqs", validation_alias=_env("QEDGE_CRYPTO_BACKEND")
    )
    key_dir: Path | None = Field(None, validation_alias=_env("QEDGE_KEY_DIR"))

    @field_validator("jwt_secret")
    @classmethod
    def _secret_is_long_enough(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value().encode()) < MIN_JWT_SECRET_BYTES:
            raise ValueError(f"JWT_SECRET_KEY must be at least {MIN_JWT_SECRET_BYTES} bytes")
        return value
