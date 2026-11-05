"""Add review resolution and claim status columns."""

from alembic import op
import sqlalchemy as sa

revision = "e2f3a4b5c6d7"
down_revision = "d1e2f3a4b5c6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "chain_posts",
        sa.Column("review_resolution", sa.String(32), nullable=True),
    )
    op.add_column(
        "chain_posts",
        sa.Column("review_resolved_by", sa.String(128), nullable=True),
    )
    op.add_column(
        "chain_posts",
        sa.Column("review_resolved_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "chain_posts",
        sa.Column("review_notes", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("chain_posts", "review_notes")
    op.drop_column("chain_posts", "review_resolved_at")
    op.drop_column("chain_posts", "review_resolved_by")
    op.drop_column("chain_posts", "review_resolution")
