"""Repository for off-network discovery tables."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from psycopg2 import extras

from app.core.database import get_db_connection
from app.discovery.identity import identity_hash_from_x_handle, resolve_identity_hash
from app.discovery.lifecycle import (
    AssetLifecycleState,
    LifecycleError,
    LifecycleEvent,
    parse_callback_event,
    transition,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DiscoveryStore:
    def upsert_source(
        self,
        *,
        source_id: str,
        adapter_type: str,
        domain: str,
        trust_score: float,
        enabled: bool,
        config: dict | None = None,
    ) -> str:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO discovery_sources (
                        id, adapter_type, domain, trust_score, enabled, config, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW())
                    ON CONFLICT (id) DO UPDATE SET
                        adapter_type = EXCLUDED.adapter_type,
                        domain = EXCLUDED.domain,
                        trust_score = EXCLUDED.trust_score,
                        enabled = EXCLUDED.enabled,
                        config = EXCLUDED.config,
                        updated_at = NOW()
                    RETURNING id::text
                    """,
                    (
                        source_id,
                        adapter_type,
                        domain,
                        trust_score,
                        enabled,
                        json.dumps(config or {}),
                    ),
                )
                row = cur.fetchone()
                conn.commit()
                return str(row[0])

    def touch_source_polled(self, source_id: str) -> None:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE discovery_sources SET last_polled_at = NOW(), updated_at = NOW() WHERE id = %s",
                    (source_id,),
                )
                conn.commit()

    def resolve_or_create_candidate(
        self,
        *,
        creator_x_handle: str | None,
        creator_confidence: float = 0.0,
    ) -> str | None:
        if not creator_x_handle:
            return None
        handle = creator_x_handle.strip().lstrip("@").lower()
        identity_hash = identity_hash_from_x_handle(handle)
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id::text FROM creator_candidates WHERE primary_x_handle = %s",
                    (handle,),
                )
                row = cur.fetchone()
                if row:
                    cur.execute(
                        """
                        UPDATE creator_candidates
                        SET identity_hash = %s,
                            creator_confidence = GREATEST(creator_confidence, %s),
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (identity_hash, creator_confidence, row[0]),
                    )
                    conn.commit()
                    return str(row[0])
                candidate_id = str(uuid.uuid4())
                cur.execute(
                    """
                    INSERT INTO creator_candidates (
                        id, primary_x_handle, identity_hash, creator_confidence, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, NOW(), NOW())
                    """,
                    (candidate_id, handle, identity_hash, creator_confidence),
                )
                conn.commit()
                return candidate_id

    def upsert_asset(
        self,
        *,
        source_id: str | None,
        external_source_url: str,
        media_type: str,
        content_kind: str = "media",
        trust_score: float,
        creator_x_handle: str | None = None,
        creator_confidence: float = 0.0,
        metadata: dict | None = None,
        priority_score: int = 0,
    ) -> tuple[str, bool]:
        candidate_id = self.resolve_or_create_candidate(
            creator_x_handle=creator_x_handle,
            creator_confidence=creator_confidence,
        )
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO discovery_assets (
                        id, source_id, external_source_url, canonical_metadata, media_type,
                        content_kind, lifecycle_state, source_trust_score, creator_confidence,
                        creator_candidate_id, priority_score, discovered_at, updated_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, 'discovered', %s, %s, %s, %s, NOW(), NOW()
                    )
                    ON CONFLICT (external_source_url) DO NOTHING
                    RETURNING id::text
                    """,
                    (
                        str(uuid.uuid4()),
                        source_id,
                        external_source_url,
                        json.dumps(metadata or {}),
                        media_type,
                        content_kind,
                        trust_score,
                        creator_confidence,
                        candidate_id,
                        priority_score,
                    ),
                )
                inserted = cur.fetchone()
                if inserted:
                    asset_id = str(inserted[0])
                    self._transition_asset_locked(cur, asset_id, LifecycleEvent.NORMALIZE)
                    if candidate_id:
                        cur.execute(
                            """
                            INSERT INTO creator_candidate_assets (
                                creator_candidate_id, discovery_asset_id, attribution_confidence
                            ) VALUES (%s, %s, %s)
                            ON CONFLICT DO NOTHING
                            """,
                            (candidate_id, asset_id, creator_confidence),
                        )
                    conn.commit()
                    return asset_id, True
                cur.execute(
                    "SELECT id::text FROM discovery_assets WHERE external_source_url = %s",
                    (external_source_url,),
                )
                row = cur.fetchone()
                conn.commit()
                return str(row[0]), False

    def get_asset(self, asset_id: str) -> dict | None:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM discovery_assets WHERE id = %s", (asset_id,))
                row = cur.fetchone()
                return dict(row) if row else None

    def asset_lifecycle_state(self, asset_id: str) -> AssetLifecycleState | None:
        asset = self.get_asset(asset_id)
        if not asset:
            return None
        raw = asset.get("lifecycle_state") or "discovered"
        try:
            return AssetLifecycleState(raw)
        except ValueError:
            return None

    def _transition_asset_locked(self, cur, asset_id: str, event: LifecycleEvent) -> AssetLifecycleState:
        cur.execute(
            "SELECT lifecycle_state FROM discovery_assets WHERE id = %s FOR UPDATE",
            (asset_id,),
        )
        row = cur.fetchone()
        if not row:
            raise ValueError(f"discovery asset not found: {asset_id}")
        current = AssetLifecycleState(row[0])
        next_state = transition(current, event)
        cur.execute(
            """
            UPDATE discovery_assets
            SET lifecycle_state = %s, updated_at = NOW()
            WHERE id = %s
            """,
            (next_state.value, asset_id),
        )
        return next_state

    def transition_asset(self, asset_id: str, event: LifecycleEvent | str) -> AssetLifecycleState:
        if isinstance(event, str):
            parsed = parse_callback_event(event)
            if parsed is None:
                try:
                    parsed = LifecycleEvent(event)
                except ValueError as exc:
                    raise ValueError(f"unknown lifecycle event: {event}") from exc
            event = parsed
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                try:
                    next_state = self._transition_asset_locked(cur, asset_id, event)
                except LifecycleError:
                    conn.rollback()
                    raise
                conn.commit()
                return next_state

    def update_embed_result(
        self,
        asset_id: str,
        *,
        work_confidence: float,
        embedding_version: str,
        identity_hash: str | None = None,
    ) -> None:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE discovery_assets
                    SET work_confidence = %s,
                        active_embedding_version = %s,
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (work_confidence, embedding_version, asset_id),
                )
                if identity_hash:
                    cur.execute(
                        """
                        UPDATE creator_candidates cc
                        SET identity_hash = COALESCE(cc.identity_hash, %s),
                            updated_at = NOW()
                        FROM discovery_assets ca
                        WHERE ca.creator_candidate_id = cc.id AND ca.id = %s
                        """,
                        (identity_hash, asset_id),
                    )
                conn.commit()

    def has_active_embed_job(self, asset_id: str) -> bool:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 1 FROM discovery_jobs
                    WHERE discovery_asset_id = %s
                      AND job_type = 'embed_asset'
                      AND status IN ('pending', 'processing')
                    LIMIT 1
                    """,
                    (asset_id,),
                )
                return cur.fetchone() is not None

    def enqueue_embed_job(
        self,
        asset_id: str,
        *,
        priority_score: int = 0,
        max_attempts: int = 5,
    ) -> str | None:
        state = self.asset_lifecycle_state(asset_id)
        if state is None:
            return None
        if state.is_at_least_indexed():
            return None
        if state.is_embed_in_progress() and self.has_active_embed_job(asset_id):
            return None
        if self.has_active_embed_job(asset_id):
            return None
        if state and state.needs_embed_enqueue():
            self.transition_asset(asset_id, LifecycleEvent.ENQUEUE)
        job_id = str(uuid.uuid4())
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO discovery_jobs (
                        id, job_type, discovery_asset_id, priority_score, status,
                        max_attempts, payload, created_at, updated_at
                    ) VALUES (%s, 'embed_asset', %s, %s, 'pending', %s, '{}', NOW(), NOW())
                    """,
                    (job_id, asset_id, priority_score, max_attempts),
                )
                conn.commit()
        return job_id

    def claim_next_embed_job(self) -> dict | None:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    UPDATE discovery_jobs
                    SET status = 'processing',
                        attempts = attempts + 1,
                        updated_at = NOW()
                    WHERE id = (
                        SELECT id FROM discovery_jobs
                        WHERE status = 'pending'
                          AND job_type = 'embed_asset'
                          AND run_after <= NOW()
                          AND attempts < max_attempts
                        ORDER BY priority_score DESC, created_at ASC
                        FOR UPDATE SKIP LOCKED
                        LIMIT 1
                    )
                    RETURNING *
                    """,
                )
                row = cur.fetchone()
                conn.commit()
                return dict(row) if row else None

    def complete_job(self, job_id: str, *, status: str, error: str | None = None) -> None:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE discovery_jobs
                    SET status = %s, last_error = %s, updated_at = NOW()
                    WHERE id = %s
                    """,
                    (status, error, job_id),
                )
                conn.commit()

    def defer_job(self, job_id: str, *, delay_seconds: int, error: str) -> None:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE discovery_jobs
                    SET status = 'pending',
                        last_error = %s,
                        run_after = %s,
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (error, _utcnow() + timedelta(seconds=delay_seconds), job_id),
                )
                conn.commit()

    def asset_counts(self) -> dict[str, int]:
        sql = """
        SELECT lifecycle_state, COUNT(*)::int AS cnt
        FROM discovery_assets
        GROUP BY lifecycle_state
        """
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                return {row[0]: row[1] for row in cur.fetchall()}

    def list_assets_by_state(self, state: str, *, limit: int = 20) -> list[dict]:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM discovery_assets
                    WHERE lifecycle_state = %s
                    ORDER BY updated_at DESC
                    LIMIT %s
                    """,
                    (state, limit),
                )
                return [dict(r) for r in cur.fetchall()]

    def asset_for_embed(self, asset_id: str) -> dict[str, Any] | None:
        asset = self.get_asset(asset_id)
        if not asset:
            return None
        candidate_handle = None
        candidate_id = asset.get("creator_candidate_id")
        if candidate_id:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT primary_x_handle FROM creator_candidates WHERE id = %s",
                        (str(candidate_id),),
                    )
                    row = cur.fetchone()
                    if row:
                        candidate_handle = row[0]
        meta = asset.get("canonical_metadata") or {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        return {
            "discovery_asset_id": str(asset["id"]),
            "external_source_url": asset["external_source_url"],
            "media_type": asset["media_type"],
            "creator_x_handle": candidate_handle or meta.get("creator_x_handle"),
            "creator_confidence": float(asset.get("creator_confidence") or 0),
            "creator_candidate_id": str(candidate_id) if candidate_id else None,
        }
