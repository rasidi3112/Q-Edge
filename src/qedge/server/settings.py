"""Server configuration read from environment variables (pydantic-settings)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from qedge.server.db import default_database_url

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
    cors_origins: tuple[str, ...] = Field(
        ("http://localhost:3000",), validation_alias=_env("CORS_ORIGINS")
    )
    debug: bool = Field(False, validation_alias=_env("DEBUG_MODE"))
    crypto_backend: Literal["liboqs", "pure"] = Field(
        "liboqs", validation_alias=_env("QEDGE_CRYPTO_BACKEND")
    )
    key_dir: Path | None = Field(None, validation_alias=_env("QEDGE_KEY_DIR"))
    data_dir: Path = Field(Path("qedge-data"), validation_alias=_env("QEDGE_DATA_DIR"))
    database_url: str | None = Field(None, validation_alias=_env("QEDGE_DATABASE_URL"))
    max_samples: int = Field(1_000_000, ge=1, validation_alias=_env("QEDGE_MAX_SAMPLES"))
    max_update_bytes: int = Field(
        16 * 1024 * 1024, ge=1024, validation_alias=_env("QEDGE_MAX_UPDATE_BYTES")
    )
    session_ttl_seconds: float = Field(
        3600.0, gt=0, validation_alias=_env("QEDGE_SESSION_TTL_SECONDS")
    )
    deadline_poll_seconds: float = Field(
        1.0, gt=0, validation_alias=_env("QEDGE_DEADLINE_POLL_SECONDS")
    )
    events_poll_seconds: float = Field(
        0.5, gt=0, validation_alias=_env("QEDGE_EVENTS_POLL_SECONDS")
    )

    def resolved_database_url(self) -> str:
        """``database_url``, or a SQLite file ``qedge.db`` under ``data_dir``."""
        if self.database_url:
            return self.database_url
        return default_database_url(self.data_dir)

    @field_validator("jwt_secret")
    @classmethod
    def _secret_is_long_enough(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value().encode()) < MIN_JWT_SECRET_BYTES:
            raise ValueError(f"JWT_SECRET_KEY must be at least {MIN_JWT_SECRET_BYTES} bytes")
        return value
