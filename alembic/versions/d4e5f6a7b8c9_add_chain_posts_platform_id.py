"""Add chain_posts.platform_id for adapter routing.

Revision ID: d4e5f6a7b8c9
Revises: c2d3e4f5a6b7
Create Date: 2026-09-11
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, None] = "c2d3e4f5a6b7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("chain_posts", sa.Column("platform_id", sa.String(length=128), nullable=True))


def downgrade() -> None:
    op.drop_column("chain_posts", "platform_id")
