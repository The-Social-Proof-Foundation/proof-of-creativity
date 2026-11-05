"""Committed beneficiary target on media asset rights bundles."""

from alembic import op
import sqlalchemy as sa

revision = "e5f6a7b8c9d0"
down_revision = "d4e5f6a7b8c9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "media_asset_rights_bundles",
        sa.Column("target_kind", sa.SmallInteger(), nullable=False, server_default="0"),
    )
    op.add_column(
        "media_asset_rights_bundles",
        sa.Column("beneficiary_address", sa.String(128), nullable=False, server_default="0x0"),
    )
    op.add_column(
        "media_asset_rights_bundles",
        sa.Column("target_vault_id", sa.String(128), nullable=True),
    )
    op.add_column(
        "media_asset_rights_bundles",
        sa.Column("target_username", sa.String(64), nullable=True),
    )
    op.add_column(
        "media_asset_rights_bundles",
        sa.Column("identity_source", sa.SmallInteger(), nullable=True),
    )
    op.add_column(
        "media_asset_rights_bundles",
        sa.Column("identity_hash", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("media_asset_rights_bundles", "identity_hash")
    op.drop_column("media_asset_rights_bundles", "identity_source")
    op.drop_column("media_asset_rights_bundles", "target_username")
    op.drop_column("media_asset_rights_bundles", "target_vault_id")
    op.drop_column("media_asset_rights_bundles", "beneficiary_address")
    op.drop_column("media_asset_rights_bundles", "target_kind")
