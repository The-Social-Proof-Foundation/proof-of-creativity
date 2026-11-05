"""Add MediaAsset registry tables for PoC Phase 5."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "f4a5b6c7d8e9"
down_revision = "e2f3a4b5c6d7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chain_posts", sa.Column("media_asset_ids", postgresql.JSONB(), server_default="[]"))
    op.add_column("chain_posts", sa.Column("composition_status", sa.Integer(), nullable=True))
    op.add_column("chain_posts", sa.Column("monetization_status", sa.Integer(), nullable=True))

    op.create_table(
        "chain_media_assets",
        sa.Column("network", sa.String(20), nullable=False),
        sa.Column("asset_id", sa.String(128), nullable=False),
        sa.Column("content_commitment", sa.String(256)),
        sa.Column("fingerprint_commitment", sa.String(256)),
        sa.Column("media_type", sa.Integer()),
        sa.Column("originality_status", sa.Integer()),
        sa.Column("lineage_parent_id", sa.String(128)),
        sa.Column("creators", postgresql.JSONB(), server_default="[]"),
        sa.Column("rights_controllers", postgresql.JSONB(), server_default="[]"),
        sa.Column("beneficiaries", postgresql.JSONB(), server_default="[]"),
        sa.Column("beneficiary_splits", postgresql.JSONB(), server_default="[]"),
        sa.Column("rights_json", postgresql.JSONB(), server_default="{}"),
        sa.Column("rights_version", sa.Integer(), server_default="1"),
        sa.Column("economics_version", sa.Integer(), server_default="1"),
        sa.Column("linked_existing", sa.Boolean(), server_default="false"),
        sa.Column("resolve_tx_digest", sa.String(128)),
        sa.Column("request_id", sa.String(128)),
        sa.Column("metadata", postgresql.JSONB(), server_default="{}"),
        sa.Column("registered_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("network", "asset_id"),
    )
    op.create_index(
        "idx_chain_media_assets_fingerprint",
        "chain_media_assets",
        ["network", "fingerprint_commitment"],
    )

    op.create_table(
        "fingerprint_observations",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("network", sa.String(20), nullable=False),
        sa.Column("fingerprint_commitment", sa.String(256), nullable=False),
        sa.Column("content_commitment", sa.String(256)),
        sa.Column("media_asset_id", sa.String(128)),
        sa.Column("request_id", sa.String(128)),
        sa.Column("media_type", sa.Integer()),
        sa.Column("submitter", sa.String(128)),
        sa.Column("observed_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_fingerprint_observations_fp",
        "fingerprint_observations",
        ["network", "fingerprint_commitment"],
    )
    op.create_index(
        "idx_fingerprint_observations_asset",
        "fingerprint_observations",
        ["network", "media_asset_id"],
    )

    op.create_table(
        "media_asset_usages",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("network", sa.String(20), nullable=False),
        sa.Column("asset_id", sa.String(128), nullable=False),
        sa.Column("container_id", sa.String(128), nullable=False),
        sa.Column("container_type", sa.Integer(), server_default="1"),
        sa.Column("usage_class", sa.Integer(), server_default="1"),
        sa.Column("position", sa.Integer(), server_default="0"),
        sa.Column("tx_digest", sa.String(128)),
        sa.Column("observed_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_media_asset_usages_asset",
        "media_asset_usages",
        ["network", "asset_id", "observed_at"],
    )
    op.create_index(
        "idx_media_asset_usages_container",
        "media_asset_usages",
        ["network", "container_id"],
    )

    op.create_table(
        "composition_analysis_records",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("network", sa.String(20), nullable=False),
        sa.Column("post_id", sa.String(128), nullable=False),
        sa.Column("composition_status", sa.Integer()),
        sa.Column("monetization_status", sa.Integer()),
        sa.Column("analysis_json", postgresql.JSONB(), server_default="{}"),
        sa.Column("manifest_json", postgresql.JSONB()),
        sa.Column("reasoning", sa.Text()),
        sa.Column("evidence_urls", postgresql.JSONB(), server_default="[]"),
        sa.Column("contains_derivatives", sa.Boolean(), server_default="false"),
        sa.Column("contains_unresolved_assets", sa.Boolean(), server_default="false"),
        sa.Column("tx_digest", sa.String(128)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_composition_analysis_post",
        "composition_analysis_records",
        ["network", "post_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_composition_analysis_post", table_name="composition_analysis_records")
    op.drop_table("composition_analysis_records")
    op.drop_index("idx_media_asset_usages_container", table_name="media_asset_usages")
    op.drop_index("idx_media_asset_usages_asset", table_name="media_asset_usages")
    op.drop_table("media_asset_usages")
    op.drop_index("idx_fingerprint_observations_asset", table_name="fingerprint_observations")
    op.drop_index("idx_fingerprint_observations_fp", table_name="fingerprint_observations")
    op.drop_table("fingerprint_observations")
    op.drop_index("idx_chain_media_assets_fingerprint", table_name="chain_media_assets")
    op.drop_table("chain_media_assets")
    op.drop_column("chain_posts", "monetization_status")
    op.drop_column("chain_posts", "composition_status")
    op.drop_column("chain_posts", "media_asset_ids")
