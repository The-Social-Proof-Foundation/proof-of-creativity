"""Regression: video parent provenance must not recurse forever."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from app.db.discovery_repository import ProvenanceHitRepository


def test_get_embedding_provenance_video_parent_no_recursion():
    """Parent media_id with only video_frame rows returns frame provenance (no RecursionError)."""
    parent_id = "6c26ce97-9ac8-4a89-97d6-99595ed9ef0e"
    frame_meta = {
        "frame_index": 6,
        "parent_media_id": parent_id,
    }
    frame_row = (frame_meta, None, "platform", "openai-clip-vit-base-patch32-v1")

    def fake_connection():
        @contextmanager
        def _conn():
            conn = MagicMock()
            cur = MagicMock()

            def execute(sql, params=None):
                sql_l = " ".join(sql.split()).lower()
                if "kind = 'video_frame'" in sql_l or "kind = 'video_frame'" in sql.lower():
                    cur.fetchone.return_value = frame_row
                else:
                    # Direct media_id / audio lookups miss for the parent.
                    cur.fetchone.return_value = None

            cur.execute.side_effect = execute
            conn.cursor.return_value.__enter__.return_value = cur
            conn.cursor.return_value.__exit__.return_value = None
            yield conn

        return _conn()

    repo = ProvenanceHitRepository()
    with patch("app.db.discovery_repository.get_db_connection", side_effect=fake_connection):
        prov = repo.get_embedding_provenance(parent_id)

    assert prov is not None
    assert prov["corpus_scope"] == "platform"
    assert prov["metadata"]["parent_media_id"] == parent_id
    assert prov["metadata"]["frame_index"] == 6


def test_get_embedding_provenance_missing_media_returns_none():
    """No embeddings/audio/frames → None (must not re-enter get_embedding_provenance)."""

    def fake_connection():
        @contextmanager
        def _conn():
            conn = MagicMock()
            cur = MagicMock()
            cur.fetchone.return_value = None
            conn.cursor.return_value.__enter__.return_value = cur
            conn.cursor.return_value.__exit__.return_value = None
            yield conn

        return _conn()

    repo = ProvenanceHitRepository()
    with patch("app.db.discovery_repository.get_db_connection", side_effect=fake_connection):
        assert repo.get_embedding_provenance("missing-media") is None
