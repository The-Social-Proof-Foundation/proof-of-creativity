"""Add off-network discovery tables (migrated from discovery-service schema)."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "f3a4b5c6d7e8"
down_revision: Union[str, None] = "e2f3a4b5c6d7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "discovery_sources",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("adapter_type", sa.String(64), nullable=False),
        sa.Column("domain", sa.String(32), server_default="creative", nullable=False),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("config", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("trust_score", sa.Float(), server_default="0.5", nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default="true", nullable=False),
        sa.Column("terms_notes", sa.Text(), nullable=True),
        sa.Column("last_polled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "creator_candidates",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("primary_x_handle", sa.String(256), nullable=True),
        sa.Column("identity_hash", sa.String(128), nullable=True),
        sa.Column("display_name", sa.String(256), nullable=True),
        sa.Column("aliases", postgresql.JSONB(), server_default="[]", nullable=False),
        sa.Column("platform_handles", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("source_urls", postgresql.JSONB(), server_default="[]", nullable=False),
        sa.Column("creator_confidence", sa.Float(), server_default="0", nullable=False),
        sa.Column("work_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("blockchain_hit_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("similarity_hit_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("lifecycle_state", sa.String(32), server_default="unresolved", nullable=False),
        sa.Column("merge_target_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["merge_target_id"], ["creator_candidates.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("primary_x_handle"),
    )

    op.create_table(
        "discovery_assets",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("external_source_url", sa.Text(), nullable=False),
        sa.Column("canonical_metadata", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("media_type", sa.String(32), nullable=False),
        sa.Column("content_kind", sa.String(16), server_default="media", nullable=False),
        sa.Column("content_hash", sa.String(128), nullable=True),
        sa.Column("metadata_hash", sa.String(128), nullable=True),
        sa.Column("lifecycle_state", sa.String(32), server_default="discovered", nullable=False),
        sa.Column("source_trust_score", sa.Float(), server_default="0.5", nullable=False),
        sa.Column("work_confidence", sa.Float(), server_default="0", nullable=False),
        sa.Column("creator_confidence", sa.Float(), server_default="0", nullable=False),
        sa.Column("creator_candidate_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("active_embedding_version", sa.String(64), nullable=True),
        sa.Column("related_on_chain_post", sa.String(128), nullable=True),
        sa.Column("priority_score", sa.BigInteger(), server_default="0", nullable=False),
        sa.Column("discovered_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("exclusion_reason", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["source_id"], ["discovery_sources.id"]),
        sa.ForeignKeyConstraint(["creator_candidate_id"], ["creator_candidates.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("external_source_url"),
    )
    op.create_index(
        "idx_discovery_assets_lifecycle",
        "discovery_assets",
        ["lifecycle_state", sa.text("priority_score DESC")],
    )
    op.create_index("idx_discovery_assets_creator", "discovery_assets", ["creator_candidate_id"])

    op.create_table(
        "creator_candidate_assets",
        sa.Column("creator_candidate_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("discovery_asset_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("attribution_confidence", sa.Float(), server_default="0", nullable=False),
        sa.Column("attribution_source", sa.String(64), server_default="auto", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["creator_candidate_id"], ["creator_candidates.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["discovery_asset_id"], ["discovery_assets.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("creator_candidate_id", "discovery_asset_id"),
    )

    op.create_table(
        "discovery_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("job_type", sa.String(32), nullable=False),
        sa.Column("discovery_asset_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("priority_score", sa.BigInteger(), server_default="0", nullable=False),
        sa.Column("status", sa.String(32), server_default="pending", nullable=False),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("max_attempts", sa.Integer(), server_default="5", nullable=False),
        sa.Column("run_after", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("payload", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["discovery_asset_id"], ["discovery_assets.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_discovery_jobs_claim",
        "discovery_jobs",
        ["status", "run_after", sa.text("priority_score DESC"), "created_at"],
    )

    op.create_foreign_key(
        "fk_provenance_hits_discovery_asset",
        "provenance_hits",
        "discovery_assets",
        ["discovery_asset_id"],
        ["id"],
    )


def downgrade() -> None:
    op.drop_constraint("fk_provenance_hits_discovery_asset", "provenance_hits", type_="foreignkey")
    op.drop_index("idx_discovery_jobs_claim", table_name="discovery_jobs")
    op.drop_table("discovery_jobs")
    op.drop_table("creator_candidate_assets")
    op.drop_index("idx_discovery_assets_creator", table_name="discovery_assets")
    op.drop_index("idx_discovery_assets_lifecycle", table_name="discovery_assets")
    op.drop_table("discovery_assets")
    op.drop_table("creator_candidates")
    op.drop_table("discovery_sources")
