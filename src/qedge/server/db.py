"""Database models, engine setup and Alembic migrations.

SQLite is the default (one file under the data directory); Postgres works
through ``QEDGE_DATABASE_URL``. Timestamps are Unix seconds stored as floats,
which avoids SQLite's lack of time zones.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import (
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

MIGRATIONS = Path(__file__).parent / "migrations"


def default_database_url(data_dir: Path) -> str:
    """SQLite URL for ``qedge.db`` inside ``data_dir``."""
    return f"sqlite:///{(data_dir / 'qedge.db').resolve()}"


class Base(DeclarativeBase):
    """Declarative base for all tables."""


class Client(Base):
    """An enrolled client or admin and its ML-DSA public key."""

    __tablename__ = "clients"

    id: Mapped[str] = mapped_column(String(256), primary_key=True)
    public_key: Mapped[bytes] = mapped_column(LargeBinary)
    scope: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[float] = mapped_column(Float)


class Challenge(Base):
    """A single-use login nonce."""

    __tablename__ = "challenges"

    nonce: Mapped[bytes] = mapped_column(LargeBinary(32), primary_key=True)
    client_id: Mapped[str] = mapped_column(ForeignKey("clients.id", ondelete="CASCADE"))
    expires_at: Mapped[float] = mapped_column(Float, index=True)


class Round(Base):
    """One federated round: its model spec, trigger rule and outcome.

    ``state`` moves ``open`` -> ``aggregating`` -> ``closed`` (or ``failed``).
    """

    __tablename__ = "rounds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    state: Mapped[str] = mapped_column(String(16), index=True)
    spec_json: Mapped[str] = mapped_column(Text)
    spec_sha256: Mapped[str] = mapped_column(String(64))
    min_clients: Mapped[int] = mapped_column(Integer)
    deadline: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[float] = mapped_column(Float)
    closed_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class Submission(Base):
    """One accepted client update; at most one per client and round."""

    __tablename__ = "submissions"
    __table_args__ = (UniqueConstraint("round_id", "client_id", name="uq_submission_round_client"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    round_id: Mapped[int] = mapped_column(ForeignKey("rounds.id", ondelete="CASCADE"), index=True)
    client_id: Mapped[str] = mapped_column(ForeignKey("clients.id", ondelete="CASCADE"))
    n_samples: Mapped[int] = mapped_column(Integer)
    local_loss: Mapped[float] = mapped_column(Float)
    weights_sha256: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[float] = mapped_column(Float)


class BlobStore:
    """Content-addressed files: each blob lives at ``root/<sha256>``.

    Args:
        root: Directory for blobs; created if missing.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def put(self, data: bytes) -> str:
        """Store ``data`` and return its hex SHA-256."""
        digest = hashlib.sha256(data).hexdigest()
        path = self.root / digest
        if not path.exists():
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
        return digest

    def get(self, digest: str) -> bytes:
        """Return the blob for ``digest``.

        Raises:
            FileNotFoundError: If no blob has that digest.
            ValueError: If the stored bytes no longer hash to ``digest``.
        """
        data = (self.root / digest).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"Blob {digest} is corrupted")
        return data


def _enable_sqlite_foreign_keys(engine: Engine) -> None:
    if engine.dialect.name != "sqlite":
        return

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_connection: object, _record: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


class Database:
    """Engine, session factory and migrations for one database URL.

    Args:
        url: SQLAlchemy URL, for example ``sqlite:////var/lib/qedge/qedge.db``.
    """

    def __init__(self, url: str) -> None:
        self.url = url
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self.engine = create_engine(url, connect_args=connect_args)
        _enable_sqlite_foreign_keys(self.engine)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    def upgrade(self) -> None:
        """Apply all Alembic migrations up to ``head``."""
        config = Config()
        config.set_main_option("script_location", str(MIGRATIONS))
        with self.engine.begin() as connection:
            config.attributes["connection"] = connection
            command.upgrade(config, "head")

    def session(self) -> Session:
        """Open a new ORM session; use it as a context manager."""
        return self.sessions()

    def dispose(self) -> None:
        """Close all pooled connections."""
        self.engine.dispose()
