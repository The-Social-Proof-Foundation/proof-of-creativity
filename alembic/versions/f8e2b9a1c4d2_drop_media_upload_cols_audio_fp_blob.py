"""drop media_files upload_user_id/upload_ip; ensure audio_fingerprints.fingerprint_data

Revision ID: f8e2b9a1c4d2
Revises: 88c1aab2dde3
Create Date: 2026-05-01

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "f8e2b9a1c4d2"
down_revision: Union[str, None] = "88c1aab2dde3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column("media_files", "upload_user_id")
    op.drop_column("media_files", "upload_ip")
    op.execute(
        sa.text(
            "ALTER TABLE audio_fingerprints "
            "ADD COLUMN IF NOT EXISTS fingerprint_data BYTEA"
        )
    )


def downgrade() -> None:
    op.add_column(
        "media_files",
        sa.Column("upload_user_id", sa.String(length=100), nullable=True),
    )
    op.add_column(
        "media_files",
        sa.Column("upload_ip", postgresql.INET(), nullable=True),
    )
