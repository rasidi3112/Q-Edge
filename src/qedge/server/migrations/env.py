"""Alembic environment: runs migrations on the connection `Database.upgrade` passes in."""

from __future__ import annotations

from alembic import context

from qedge.server.db import Base

connection = context.config.attributes["connection"]
context.configure(connection=connection, target_metadata=Base.metadata, render_as_batch=True)
with context.begin_transaction():
    context.run_migrations()
