"""Rounds and submissions.

Revision ID: 0002
Revises: 0001
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the rounds and submissions tables."""
    op.create_table(
        "rounds",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("spec_json", sa.Text(), nullable=False),
        sa.Column("spec_sha256", sa.String(64), nullable=False),
        sa.Column("min_clients", sa.Integer(), nullable=False),
        sa.Column("deadline", sa.Float(), nullable=True),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("closed_at", sa.Float(), nullable=True),
        sa.Column("model_sha256", sa.String(64), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
    )
    op.create_index("ix_rounds_state", "rounds", ["state"])
    op.create_table(
        "submissions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "round_id", sa.Integer(), sa.ForeignKey("rounds.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column(
            "client_id",
            sa.String(256),
            sa.ForeignKey("clients.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("n_samples", sa.Integer(), nullable=False),
        sa.Column("local_loss", sa.Float(), nullable=False),
        sa.Column("weights_sha256", sa.String(64), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.UniqueConstraint("round_id", "client_id", name="uq_submission_round_client"),
    )
    op.create_index("ix_submissions_round_id", "submissions", ["round_id"])


def downgrade() -> None:
    """Drop the rounds and submissions tables."""
    op.drop_index("ix_submissions_round_id", "submissions")
    op.drop_table("submissions")
    op.drop_index("ix_rounds_state", "rounds")
    op.drop_table("rounds")
