"""Phase 2–5 oracle tables: pending derivatives, graph edges, discovery proposals."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "a9b8c7d6e5f4"
down_revision = "f4a5b6c7d8e9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chain_media_assets", sa.Column("asset_kind", sa.Integer(), server_default="0"))
    op.add_column("chain_media_assets", sa.Column("pending_id", sa.String(128), nullable=True))
    op.add_column("chain_media_assets", sa.Column("policy_version", sa.Integer(), server_default="0"))

    op.create_table(
        "pending_derivative_assets",
        sa.Column("network", sa.String(20), nullable=False),
        sa.Column("pending_id", sa.String(128), nullable=False),
        sa.Column("request_id", sa.String(128)),
        sa.Column("content_commitment", sa.String(256)),
        sa.Column("fingerprint_commitment", sa.String(256)),
        sa.Column("media_type", sa.Integer()),
        sa.Column("asset_kind", sa.Integer(), server_default="0"),
        sa.Column("creator", sa.String(128)),
        sa.Column("status", sa.String(32), server_default="pending"),
        sa.Column("finalize_tx_digest", sa.String(128)),
        sa.Column("child_asset_id", sa.String(128)),
        sa.Column("metadata", postgresql.JSONB(), server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("network", "pending_id"),
    )
    op.create_index("idx_pending_derivative_request", "pending_derivative_assets", ["network", "request_id"])

    op.create_table(
        "derivative_edges",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("network", sa.String(20), nullable=False),
        sa.Column("parent_asset_id", sa.String(128), nullable=False),
        sa.Column("child_asset_id", sa.String(128), nullable=False),
        sa.Column("relationship_type", sa.Integer(), server_default="1"),
        sa.Column("license_instance_id", sa.String(128)),
        sa.Column("template_version_id", sa.String(128)),
        sa.Column("parent_share_bps", sa.Integer()),
        sa.Column("tx_digest", sa.String(128)),
        sa.Column("observed_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_derivative_edges_child", "derivative_edges", ["network", "child_asset_id"])
    op.create_index("idx_derivative_edges_parent", "derivative_edges", ["network", "parent_asset_id"])

    op.create_table(
        "detected_asset_relationships",
        sa.Column("network", sa.String(20), nullable=False),
        sa.Column("proposal_id", sa.String(128), nullable=False),
        sa.Column("accused_pending_id", sa.String(128), nullable=False),
        sa.Column("accused_asset_id", sa.String(128)),
        sa.Column("original_asset_id", sa.String(128), nullable=False),
        sa.Column("similarity_bps", sa.Integer(), nullable=False),
        sa.Column("status", sa.Integer(), server_default="0"),
        sa.Column("evidence_commitment", sa.String(256)),
        sa.Column("tx_digest", sa.String(128)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("network", "proposal_id"),
    )
    op.create_index(
        "idx_detected_relationships_pending",
        "detected_asset_relationships",
        ["network", "accused_pending_id"],
    )

    op.create_table(
        "post_enforcement_snapshots",
        sa.Column("network", sa.String(20), nullable=False),
        sa.Column("post_id", sa.String(128), nullable=False),
        sa.Column("bindings_json", postgresql.JSONB(), server_default="[]"),
        sa.Column("usage_decisions_json", postgresql.JSONB(), server_default="[]"),
        sa.Column("usage_denials_json", postgresql.JSONB(), server_default="[]"),
        sa.Column("playback_policy_json", postgresql.JSONB(), server_default="{}"),
        sa.Column("tx_digest", sa.String(128)),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("network", "post_id"),
    )


def downgrade() -> None:
    op.drop_table("post_enforcement_snapshots")
    op.drop_index("idx_detected_relationships_pending", table_name="detected_asset_relationships")
    op.drop_table("detected_asset_relationships")
    op.drop_index("idx_derivative_edges_parent", table_name="derivative_edges")
    op.drop_index("idx_derivative_edges_child", table_name="derivative_edges")
    op.drop_table("derivative_edges")
    op.drop_index("idx_pending_derivative_request", table_name="pending_derivative_assets")
    op.drop_table("pending_derivative_assets")
    op.drop_column("chain_media_assets", "policy_version")
    op.drop_column("chain_media_assets", "pending_id")
    op.drop_column("chain_media_assets", "asset_kind")
