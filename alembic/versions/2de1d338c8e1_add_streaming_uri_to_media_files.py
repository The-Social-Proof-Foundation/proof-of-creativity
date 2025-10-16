"""add_streaming_uri_to_media_files

Revision ID: 2de1d338c8e1
Revises: ce3d214088e3
Create Date: 2025-10-16 07:06:03.670611

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2de1d338c8e1'
down_revision: Union[str, None] = 'ce3d214088e3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add streaming_uri column to media_files table for Cloudflare Stream support."""
    # Add streaming_uri column (nullable, opt-in for videos)
    op.add_column('media_files', 
        sa.Column('streaming_uri', sa.Text(), nullable=True,
                 comment='Cloudflare Stream URI for video streaming (optional, opt-in per upload)')
    )


def downgrade() -> None:
    """Remove streaming_uri column."""
    op.drop_column('media_files', 'streaming_uri')
