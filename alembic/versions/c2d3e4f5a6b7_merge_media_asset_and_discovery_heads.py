"""Merge MediaAsset branch and discovery/stream_id branch.

Revision ID: c2d3e4f5a6b7
Revises: b1c2d3e4f5a6, a1b2c3d4e5f6
Create Date: 2026-08-28

Those two heads forked from e2f3a4b5c6d7. upgrade head only stamped one, so
chain_posts never received media_asset_ids.
"""

from typing import Sequence, Union

revision: str = "c2d3e4f5a6b7"
down_revision: Union[str, tuple[str, ...], None] = ("b1c2d3e4f5a6", "a1b2c3d4e5f6")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
