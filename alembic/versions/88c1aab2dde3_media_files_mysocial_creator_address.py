"""media_files optional creator_address for PoC attribution

Revision ID: 88c1aab2dde3
Revises: 2de1d338c8e1
Create Date: 2026-04-30

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "88c1aab2dde3"
down_revision: Union[str, None] = "2de1d338c8e1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "media_files",
        sa.Column("creator_address", sa.String(length=128), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("media_files", "creator_address")
