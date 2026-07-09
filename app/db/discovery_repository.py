"""Repository for local provenance hit audit rows."""

from __future__ import annotations

import uuid
from typing import Any

from app.core.database import get_db_connection


class ProvenanceHitRepository:
    def record(
        self,
        *,
        network: str,
        post_id: str,
        query_media_id: str | None,
        discovery_asset_id: str | None,
        creator_candidate_id: str | None,
        similarity_score: float,
        match_type: str | None,
        work_confidence: float,
        creator_confidence: float,
        decision: str,
        vault_provisioned: bool = False,
        vault_identity_hash: str | None = None,
    ) -> str:
        hit_id = str(uuid.uuid4())
        sql = """
        INSERT INTO provenance_hits (
            id, network, post_id, query_media_id, discovery_asset_id, creator_candidate_id,
            similarity_score, match_type, work_confidence, creator_confidence,
            decision, vault_provisioned, vault_identity_hash
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        hit_id,
                        network,
                        post_id,
                        query_media_id,
                        discovery_asset_id,
                        creator_candidate_id,
                        similarity_score,
                        match_type,
                        work_confidence,
                        creator_confidence,
                        decision,
                        vault_provisioned,
                        vault_identity_hash,
                    ),
                )
                conn.commit()
        return hit_id

    def get_embedding_provenance(self, media_id: str) -> dict[str, Any] | None:
        """Resolve provenance from embeddings, audio fingerprints, or video parent rows."""
        provenance = self._provenance_from_embeddings(media_id)
        if provenance:
            return provenance
        provenance = self._provenance_from_audio_fingerprints(media_id)
        if provenance:
            return provenance
        return self._provenance_from_video_parent(media_id)

    def _normalize_provenance_row(
        self,
        metadata: Any,
        discovery_asset_id: Any,
        corpus_scope: Any,
        embedding_version: Any,
    ) -> dict[str, Any]:
        meta = metadata if isinstance(metadata, dict) else {}
        return {
            "metadata": meta,
            "discovery_asset_id": (
                str(discovery_asset_id) if discovery_asset_id else meta.get("discovery_asset_id")
            ),
            "corpus_scope": corpus_scope or meta.get("corpus_scope", "platform"),
            "embedding_version": embedding_version or meta.get("embedding_version"),
        }

    def _provenance_from_embeddings(self, media_id: str) -> dict[str, Any] | None:
        sql = """
        SELECT metadata, discovery_asset_id, corpus_scope, embedding_version
        FROM media_embeddings
        WHERE media_id = %s
        ORDER BY uploaded_at DESC
        LIMIT 1
        """
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (media_id,))
                row = cur.fetchone()
        if not row:
            return None
        return self._normalize_provenance_row(*row)

    def _provenance_from_audio_fingerprints(self, media_id: str) -> dict[str, Any] | None:
        sql = """
        SELECT discovery_asset_id, corpus_scope, embedding_version
        FROM audio_fingerprints
        WHERE media_id = %s
        ORDER BY created_at DESC
        LIMIT 1
        """
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (media_id,))
                row = cur.fetchone()
        if not row:
            return None
        discovery_asset_id, corpus_scope, embedding_version = row
        embedding_prov = self._provenance_from_embeddings(media_id)
        meta = (embedding_prov or {}).get("metadata") or {}
        return {
            "metadata": meta,
            "discovery_asset_id": (
                str(discovery_asset_id) if discovery_asset_id else meta.get("discovery_asset_id")
            ),
            "corpus_scope": corpus_scope or meta.get("corpus_scope", "platform"),
            "embedding_version": embedding_version or meta.get("embedding_version"),
        }

    def _provenance_from_video_parent(self, media_id: str) -> dict[str, Any] | None:
        if "_frame_" in media_id:
            parent_id = media_id.rsplit("_frame_", 1)[0]
            return self.get_embedding_provenance(parent_id)
        sql = """
        SELECT metadata, discovery_asset_id, corpus_scope, embedding_version
        FROM media_embeddings
        WHERE kind = 'video_frame'
          AND metadata->>'parent_media_id' = %s
        ORDER BY uploaded_at DESC
        LIMIT 1
        """
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (media_id,))
                row = cur.fetchone()
        if not row:
            return self.get_embedding_provenance(media_id)
        frame_prov = self._normalize_provenance_row(*row)
        parent_id = (frame_prov.get("metadata") or {}).get("parent_media_id")
        if parent_id:
            parent_prov = self.get_embedding_provenance(str(parent_id))
            if parent_prov:
                return parent_prov
        return frame_prov
