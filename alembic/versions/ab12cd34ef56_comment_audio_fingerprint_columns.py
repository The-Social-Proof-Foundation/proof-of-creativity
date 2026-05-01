"""Comment audio_fingerprints fp_hash and fingerprint_data semantics.

Revision ID: ab12cd34ef56
Revises: 9f3a2b1c0d4e
Create Date: 2026-05-01

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "ab12cd34ef56"
down_revision: Union[str, None] = "9f3a2b1c0d4e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        sa.text(
            "COMMENT ON COLUMN audio_fingerprints.fp_hash IS "
            "'SHA-1 hex digest (40 lowercase chars): composite lookup key from constellation hashing.'"
        )
    )
    op.execute(
        sa.text(
            "COMMENT ON COLUMN audio_fingerprints.fingerprint_data IS "
            "'Pickled constellation hash tuples (BYTEA). Leading hex when previewed is pickle framing, not a short fingerprint string.'"
        )
    )


def downgrade() -> None:
    op.execute(
        sa.text(
            "COMMENT ON COLUMN audio_fingerprints.fp_hash IS "
            "'SHA-256 hash of audio fingerprint for fast lookup'"
        )
    )
    op.execute(sa.text("COMMENT ON COLUMN audio_fingerprints.fingerprint_data IS NULL"))
