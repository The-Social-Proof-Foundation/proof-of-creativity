"""Add discovery corpus scoping, embedding versioning, and provenance hits."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "d1e2f3a4b5c6"
down_revision: Union[str, None] = "c7d8e9f0a1b2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _add_corpus_columns(table: str) -> None:
    op.add_column(
        table,
        sa.Column("corpus_scope", sa.String(32), nullable=False, server_default="platform"),
    )
    op.add_column(table, sa.Column("discovery_asset_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column(table, sa.Column("embedding_model", sa.String(128), nullable=True))
    op.add_column(table, sa.Column("embedding_version", sa.String(64), nullable=True))
    op.add_column(table, sa.Column("embedding_dimension", sa.Integer(), nullable=True))
    op.add_column(
        table,
        sa.Column(
            "embedding_created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
    )


def upgrade() -> None:
    for table in ("media_embeddings", "audio_fingerprints", "image_hashes"):
        _add_corpus_columns(table)
        op.create_index(f"idx_{table}_corpus_scope", table, ["corpus_scope"])

    op.create_table(
        "provenance_hits",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("network", sa.String(20), nullable=False),
        sa.Column("post_id", sa.String(128), nullable=False),
        sa.Column("query_media_id", sa.String(100)),
        sa.Column("discovery_asset_id", postgresql.UUID(as_uuid=True)),
        sa.Column("creator_candidate_id", postgresql.UUID(as_uuid=True)),
        sa.Column("similarity_score", sa.Float(), server_default="0"),
        sa.Column("match_type", sa.String(50)),
        sa.Column("work_confidence", sa.Float(), server_default="0"),
        sa.Column("creator_confidence", sa.Float(), server_default="0"),
        sa.Column("decision", sa.String(32), server_default="pending"),
        sa.Column("vault_provisioned", sa.Boolean(), server_default="false"),
        sa.Column("vault_identity_hash", sa.String(128)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_provenance_hits_post",
        "provenance_hits",
        ["network", "post_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_provenance_hits_post", table_name="provenance_hits")
    op.drop_table("provenance_hits")
    for table in ("image_hashes", "audio_fingerprints", "media_embeddings"):
        op.drop_index(f"idx_{table}_corpus_scope", table_name=table)
        op.drop_column(table, "embedding_created_at")
        op.drop_column(table, "embedding_dimension")
        op.drop_column(table, "embedding_version")
        op.drop_column(table, "embedding_model")
        op.drop_column(table, "discovery_asset_id")
        op.drop_column(table, "corpus_scope")
