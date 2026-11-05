"""Rename attribution_records tx / wallet columns.

Revision ID: 9f3a2b1c0d4e
Revises: f8e2b9a1c4d2
Create Date: 2026-05-01

"""
from typing import Sequence, Union

from alembic import op


revision: str = "9f3a2b1c0d4e"
down_revision: Union[str, None] = "f8e2b9a1c4d2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE attribution_records RENAME COLUMN blockchain_tx_hash TO tx_hash"
    )
    op.execute(
        "ALTER TABLE attribution_records RENAME COLUMN blockchain_address TO wallet_address"
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE attribution_records RENAME COLUMN wallet_address TO blockchain_address"
    )
    op.execute(
        "ALTER TABLE attribution_records RENAME COLUMN tx_hash TO blockchain_tx_hash"
    )
