"""Media asset rights governance dispute claim bundles."""

from alembic import op
import sqlalchemy as sa

revision = "b1c2d3e4f5a6"
down_revision = "a9b8c7d6e5f4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "media_asset_rights_bundles",
        sa.Column("proposal_id", sa.String(128), nullable=False),
        sa.Column("network", sa.String(20), nullable=False),
        sa.Column("media_asset_id", sa.String(128), nullable=False),
        sa.Column("claims_commitment", sa.LargeBinary(), nullable=False),
        sa.Column("claims_bcs", sa.LargeBinary(), nullable=False),
        sa.Column("usage_grants_bcs", sa.LargeBinary(), nullable=False),
        sa.Column("submitter", sa.String(128), nullable=False),
        sa.Column("status", sa.String(32), server_default="pending", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("proposal_id"),
    )
    op.create_index(
        "idx_media_asset_rights_bundles_asset",
        "media_asset_rights_bundles",
        ["network", "media_asset_id"],
    )


def downgrade() -> None:
    op.drop_index("idx_media_asset_rights_bundles_asset", table_name="media_asset_rights_bundles")
    op.drop_table("media_asset_rights_bundles")
