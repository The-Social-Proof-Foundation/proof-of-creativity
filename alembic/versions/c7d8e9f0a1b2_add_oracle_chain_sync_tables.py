"""Add oracle / chain sync tables for multi-network PoC pipeline."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "c7d8e9f0a1b2"
down_revision: Union[str, None] = "ab12cd34ef56"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "chain_posts",
        sa.Column("network", sa.String(20), nullable=False),
        sa.Column("post_id", sa.String(128), nullable=False),
        sa.Column("creator_address", sa.String(128)),
        sa.Column("enable_poc", sa.String(10), server_default="true"),
        sa.Column("media_urls", postgresql.JSONB(), server_default="[]"),
        sa.Column("media_types", postgresql.JSONB(), server_default="[]"),
        sa.Column("analysis_status", sa.String(32), server_default="discovered"),
        sa.Column("poc_outcome", sa.Integer()),
        sa.Column("highest_similarity_score", sa.Integer()),
        sa.Column("proof_bundle_uri", sa.Text()),
        sa.Column("tx_digest", sa.String(128)),
        sa.Column("discovered_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("metadata", postgresql.JSONB(), server_default="{}"),
        sa.PrimaryKeyConstraint("network", "post_id"),
    )

    op.create_table(
        "grpc_sync_checkpoints",
        sa.Column("network", sa.String(20), nullable=False),
        sa.Column("stream_id", sa.String(64), server_default="default", nullable=False),
        sa.Column("checkpoint_sequence", sa.BigInteger(), server_default="0"),
        sa.Column("last_transaction_digest", sa.String(128)),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("network", "stream_id"),
    )

    op.create_table(
        "oracle_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("network", sa.String(20), nullable=False),
        sa.Column("post_id", sa.String(128), nullable=False),
        sa.Column("job_type", sa.String(32), nullable=False, server_default="analyze_post"),
        sa.Column("media_url", sa.Text()),
        sa.Column("media_index", sa.Integer(), server_default="0"),
        sa.Column("media_type", sa.Integer()),
        sa.Column("status", sa.String(32), server_default="pending"),
        sa.Column("attempts", sa.Integer(), server_default="0"),
        sa.Column("last_error", sa.Text()),
        sa.Column("payload", postgresql.JSONB(), server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_oracle_jobs_status_network", "oracle_jobs", ["network", "status", "created_at"])
    op.create_index("idx_oracle_jobs_post", "oracle_jobs", ["network", "post_id"])

    op.create_table(
        "chain_attestations",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("network", sa.String(20), nullable=False),
        sa.Column("post_id", sa.String(128), nullable=False),
        sa.Column("tx_digest", sa.String(128)),
        sa.Column("media_type", sa.Integer()),
        sa.Column("highest_similarity_score", sa.Integer()),
        sa.Column("original_creator", sa.String(128)),
        sa.Column("derivative_redirection_target", sa.Integer()),
        sa.Column("poc_outcome", sa.Integer()),
        sa.Column("reasoning", sa.Text()),
        sa.Column("evidence_urls", postgresql.JSONB(), server_default="[]"),
        sa.Column("status", sa.String(32), server_default="submitted"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_chain_attestations_post", "chain_attestations", ["network", "post_id", "created_at"])

    op.create_table(
        "media_post_links",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("network", sa.String(20), nullable=False),
        sa.Column("post_id", sa.String(128), nullable=False),
        sa.Column("media_id", sa.String(100), nullable=False),
        sa.Column("media_url", sa.Text()),
        sa.Column("media_index", sa.Integer(), server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_media_post_links_post", "media_post_links", ["network", "post_id"])
    op.create_index("idx_media_post_links_media", "media_post_links", ["media_id"])

    op.create_table(
        "username_beneficiaries",
        sa.Column("network", sa.String(20), nullable=False),
        sa.Column("identity_hash", sa.String(128), nullable=False),
        sa.Column("username", sa.String(256)),
        sa.Column("beneficiary_address", sa.String(128)),
        sa.Column("vault_object_id", sa.String(128)),
        sa.Column("provision_tx_digest", sa.String(128)),
        sa.Column("claimed", sa.String(10), server_default="false"),
        sa.Column("claim_tx_digest", sa.String(128)),
        sa.Column("metadata", postgresql.JSONB(), server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("network", "identity_hash"),
    )

    op.create_table(
        "chain_config_cache",
        sa.Column("network", sa.String(20), nullable=False),
        sa.Column("config_json", postgresql.JSONB(), server_default="{}"),
        sa.Column("fetched_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("network"),
    )

    op.create_table(
        "external_identities",
        sa.Column("network", sa.String(20), nullable=False),
        sa.Column("identity_hash", sa.String(128), nullable=False),
        sa.Column("platform", sa.String(64)),
        sa.Column("external_id", sa.String(256)),
        sa.Column("display_name", sa.String(256)),
        sa.Column("metadata", postgresql.JSONB(), server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("network", "identity_hash"),
    )


def downgrade() -> None:
    op.drop_table("external_identities")
    op.drop_table("chain_config_cache")
    op.drop_table("username_beneficiaries")
    op.drop_index("idx_media_post_links_media", table_name="media_post_links")
    op.drop_index("idx_media_post_links_post", table_name="media_post_links")
    op.drop_table("media_post_links")
    op.drop_index("idx_chain_attestations_post", table_name="chain_attestations")
    op.drop_table("chain_attestations")
    op.drop_index("idx_oracle_jobs_post", table_name="oracle_jobs")
    op.drop_index("idx_oracle_jobs_status_network", table_name="oracle_jobs")
    op.drop_table("oracle_jobs")
    op.drop_table("grpc_sync_checkpoints")
    op.drop_table("chain_posts")
