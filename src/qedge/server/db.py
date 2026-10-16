"""Database models, engine setup and Alembic migrations.

SQLite is the default (one file under the data directory); Postgres works
through ``QEDGE_DATABASE_URL``. Timestamps are Unix seconds stored as floats,
which avoids SQLite's lack of time zones.
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Float, ForeignKey, LargeBinary, String, create_engine, event
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
