"""Widen grpc_sync_checkpoints.stream_id for full 0x Move addresses.

Revision ID: a1b2c3d4e5f6
Revises: f3a4b5c6d7e8
Create Date: 2026-07-26

Full package / stream ids are 0x + 64 hex = 66 chars. VARCHAR(64) rejected those
when the id had no leading zeros to strip (short form still 66).
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "f3a4b5c6d7e8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "grpc_sync_checkpoints",
        "stream_id",
        existing_type=sa.String(length=64),
        type_=sa.String(length=66),
        existing_nullable=False,
        existing_server_default="default",
    )


def downgrade() -> None:
    op.alter_column(
        "grpc_sync_checkpoints",
        "stream_id",
        existing_type=sa.String(length=66),
        type_=sa.String(length=64),
        existing_nullable=False,
        existing_server_default="default",
    )
