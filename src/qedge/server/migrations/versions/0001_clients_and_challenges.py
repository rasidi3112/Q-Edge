"""Clients and login challenges.

Revision ID: 0001
Revises:
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the clients and challenges tables."""
    op.create_table(
        "clients",
        sa.Column("id", sa.String(256), primary_key=True),
        sa.Column("public_key", sa.LargeBinary(), nullable=False),
        sa.Column("scope", sa.String(16), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
    )
    op.create_table(
        "challenges",
        sa.Column("nonce", sa.LargeBinary(32), primary_key=True),
        sa.Column(
            "client_id",
            sa.String(256),
            sa.ForeignKey("clients.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.Float(), nullable=False),
    )
    op.create_index("ix_challenges_expires_at", "challenges", ["expires_at"])


def downgrade() -> None:
    """Drop the clients and challenges tables."""
    op.drop_index("ix_challenges_expires_at", "challenges")
    op.drop_table("challenges")
    op.drop_table("clients")
