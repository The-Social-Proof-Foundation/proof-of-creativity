"""Repository layer for oracle / chain sync tables."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from psycopg2 import extras

from app.core.database import get_db_connection


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ChainPostRepository:
    def upsert_discovered(
        self,
        network: str,
        post_id: str,
        *,
        creator_address: str | None,
        enable_poc: bool,
        media_urls: list[str],
        media_types: list[int] | None = None,
        media_asset_ids: list[str] | None = None,
        composition_status: int | None = None,
        monetization_status: int | None = None,
        metadata: dict | None = None,
    ) -> bool:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chain_posts (
                        network, post_id, creator_address, enable_poc, media_urls,
                        media_types, media_asset_ids, composition_status, monetization_status,
                        analysis_status, metadata, discovered_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'discovered', %s, NOW(), NOW())
                    ON CONFLICT (network, post_id) DO NOTHING
                    RETURNING post_id
                    """,
                    (
                        network,
                        post_id,
                        creator_address,
                        "true" if enable_poc else "false",
                        json.dumps(media_urls),
                        json.dumps(media_types or []),
                        json.dumps(media_asset_ids or []),
                        composition_status,
                        monetization_status,
                        json.dumps(metadata or {}),
                    ),
                )
                inserted = cur.fetchone() is not None
                conn.commit()
                return inserted

    def get(self, network: str, post_id: str) -> dict | None:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM chain_posts WHERE network = %s AND post_id = %s",
                    (network, post_id),
                )
                row = cur.fetchone()
                return dict(row) if row else None

    def update_status(
        self,
        network: str,
        post_id: str,
        *,
        analysis_status: str,
        poc_outcome: int | None = None,
        highest_similarity_score: int | None = None,
        proof_bundle_uri: str | None = None,
        tx_digest: str | None = None,
    ) -> None:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE chain_posts
                    SET analysis_status = %s,
                        poc_outcome = COALESCE(%s, poc_outcome),
                        highest_similarity_score = COALESCE(%s, highest_similarity_score),
                        proof_bundle_uri = COALESCE(%s, proof_bundle_uri),
                        tx_digest = COALESCE(%s, tx_digest),
                        updated_at = NOW()
                    WHERE network = %s AND post_id = %s
                    """,
                    (
                        analysis_status,
                        poc_outcome,
                        highest_similarity_score,
                        proof_bundle_uri,
                        tx_digest,
                        network,
                        post_id,
                    ),
                )
                conn.commit()

    def list_needs_review(
        self,
        network: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM chain_posts
                    WHERE network = %s AND analysis_status = 'needs_review'
                    ORDER BY updated_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    (network, limit, offset),
                )
                return [dict(r) for r in cur.fetchall()]

    def resolve_review(
        self,
        network: str,
        post_id: str,
        *,
        action: str,
        resolved_by: str | None = None,
        notes: str | None = None,
    ) -> None:
        status_map = {
            "approve_escrow": "pending_reanalysis",
            "reject": "rejected",
            "requeue": "discovered",
        }
        new_status = status_map.get(action, "needs_review")
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE chain_posts
                    SET analysis_status = %s,
                        review_resolution = %s,
                        review_resolved_by = %s,
                        review_notes = %s,
                        review_resolved_at = NOW(),
                        updated_at = NOW()
                    WHERE network = %s AND post_id = %s
                    """,
                    (new_status, action, resolved_by, notes, network, post_id),
                )
                conn.commit()


class CheckpointRepository:
    def get(self, network: str, stream_id: str = "default") -> dict | None:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM grpc_sync_checkpoints WHERE network = %s AND stream_id = %s",
                    (network, stream_id),
                )
                row = cur.fetchone()
                return dict(row) if row else None

    def save(
        self,
        network: str,
        checkpoint_sequence: int,
        *,
        stream_id: str = "default",
        last_transaction_digest: str | None = None,
    ) -> None:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO grpc_sync_checkpoints (
                        network, stream_id, checkpoint_sequence, last_transaction_digest, updated_at
                    ) VALUES (%s, %s, %s, %s, NOW())
                    ON CONFLICT (network, stream_id) DO UPDATE SET
                        checkpoint_sequence = EXCLUDED.checkpoint_sequence,
                        last_transaction_digest = EXCLUDED.last_transaction_digest,
                        updated_at = NOW()
                    """,
                    (network, stream_id, checkpoint_sequence, last_transaction_digest),
                )
                conn.commit()


class JobRepository:
    def enqueue(
        self,
        network: str,
        post_id: str,
        *,
        job_type: str = "analyze_post",
        media_url: str | None = None,
        media_index: int = 0,
        media_type: int | None = None,
        payload: dict | None = None,
    ) -> str:
        job_id = str(uuid.uuid4())
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO oracle_jobs (
                        id, network, post_id, job_type, media_url, media_index,
                        media_type, status, payload, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending', %s, NOW(), NOW())
                    """,
                    (
                        job_id,
                        network,
                        post_id,
                        job_type,
                        media_url,
                        media_index,
                        media_type,
                        json.dumps(payload or {}),
                    ),
                )
                conn.commit()
        return job_id

    def claim_next(self, network: str, job_type: str | None = None) -> dict | None:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                if job_type:
                    cur.execute(
                        """
                        UPDATE oracle_jobs
                        SET status = 'processing', updated_at = NOW(), attempts = attempts + 1
                        WHERE id = (
                            SELECT id FROM oracle_jobs
                            WHERE network = %s AND status = 'pending' AND job_type = %s
                              AND updated_at <= NOW()
                            ORDER BY created_at ASC
                            FOR UPDATE SKIP LOCKED
                            LIMIT 1
                        )
                        RETURNING *
                        """,
                        (network, job_type),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE oracle_jobs
                        SET status = 'processing', updated_at = NOW(), attempts = attempts + 1
                        WHERE id = (
                            SELECT id FROM oracle_jobs
                            WHERE network = %s AND status = 'pending'
                              AND updated_at <= NOW()
                            ORDER BY created_at ASC
                            FOR UPDATE SKIP LOCKED
                            LIMIT 1
                        )
                        RETURNING *
                        """,
                        (network,),
                    )
                row = cur.fetchone()
                conn.commit()
                return dict(row) if row else None

    def requeue(
        self,
        job_id: str,
        *,
        error: str | None = None,
        delay_seconds: int = 15,
    ) -> None:
        """Return job to pending after a delay (used for media-not-ready / 404 races)."""
        delay = max(1, int(delay_seconds))
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE oracle_jobs
                    SET status = 'pending',
                        last_error = %s,
                        updated_at = NOW() + (%s * INTERVAL '1 second')
                    WHERE id = %s
                    """,
                    (error, delay, job_id),
                )
                conn.commit()

    def complete(self, job_id: str, status: str = "completed", error: str | None = None) -> None:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE oracle_jobs
                    SET status = %s, last_error = %s, updated_at = NOW()
                    WHERE id = %s
                    """,
                    (status, error, job_id),
                )
                conn.commit()

    def pending_count(self, network: str) -> int:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM oracle_jobs WHERE network = %s AND status = 'pending'",
                    (network,),
                )
                row = cur.fetchone()
                return int(row[0]) if row else 0

    def get(self, job_id: str) -> dict | None:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM oracle_jobs WHERE id = %s", (job_id,))
                row = cur.fetchone()
                return dict(row) if row else None

    def has_active_job_for_proposal(
        self,
        network: str,
        job_type: str,
        proposal_id: str,
    ) -> bool:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 1 FROM oracle_jobs
                    WHERE network = %s
                      AND job_type = %s
                      AND status IN ('pending', 'processing')
                      AND payload->>'proposal_id' = %s
                    LIMIT 1
                    """,
                    (network, job_type, proposal_id),
                )
                return cur.fetchone() is not None


class AttestationRepository:
    def insert(self, record: dict) -> str:
        att_id = str(uuid.uuid4())
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chain_attestations (
                        id, network, post_id, tx_digest, media_type, highest_similarity_score,
                        original_creator, derivative_redirection_target, poc_outcome,
                        reasoning, evidence_urls, status, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    """,
                    (
                        att_id,
                        record["network"],
                        record["post_id"],
                        record.get("tx_digest"),
                        record.get("media_type"),
                        record.get("highest_similarity_score"),
                        record.get("original_creator"),
                        record.get("derivative_redirection_target"),
                        record.get("poc_outcome"),
                        record.get("reasoning"),
                        json.dumps(record.get("evidence_urls") or []),
                        record.get("status", "submitted"),
                    ),
                )
                conn.commit()
        return att_id

    def list_for_post(self, network: str, post_id: str) -> list[dict]:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM chain_attestations
                    WHERE network = %s AND post_id = %s
                    ORDER BY created_at DESC
                    """,
                    (network, post_id),
                )
                return [dict(r) for r in cur.fetchall()]

    def mark_confirmed(self, network: str, tx_digest: str) -> None:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE chain_attestations
                    SET status = 'confirmed'
                    WHERE network = %s AND tx_digest = %s
                    """,
                    (network, tx_digest),
                )
                conn.commit()

    def list_submitted(self, network: str, limit: int = 20) -> list[dict]:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM chain_attestations
                    WHERE network = %s AND status = 'submitted' AND tx_digest IS NOT NULL
                    ORDER BY created_at ASC
                    LIMIT %s
                    """,
                    (network, limit),
                )
                return [dict(r) for r in cur.fetchall()]


class MediaPostLinkRepository:
    def link(
        self,
        network: str,
        post_id: str,
        media_id: str,
        *,
        media_url: str | None = None,
        media_index: int = 0,
    ) -> None:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO media_post_links (
                        id, network, post_id, media_id, media_url, media_index, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, NOW())
                    """,
                    (str(uuid.uuid4()), network, post_id, media_id, media_url, media_index),
                )
                conn.commit()


class ConfigCacheRepository:
    def get(self, network: str) -> dict | None:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT config_json FROM chain_config_cache WHERE network = %s",
                    (network,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                cfg = row.get("config_json")
                return cfg if isinstance(cfg, dict) else json.loads(cfg or "{}")

    def set(self, network: str, config: dict) -> None:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chain_config_cache (network, config_json, fetched_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (network) DO UPDATE SET
                        config_json = EXCLUDED.config_json,
                        fetched_at = NOW()
                    """,
                    (network, json.dumps(config)),
                )
                conn.commit()


class UsernameBeneficiaryRepository:
    def get(self, network: str, identity_hash: str) -> dict | None:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM username_beneficiaries WHERE network = %s AND identity_hash = %s",
                    (network, identity_hash),
                )
                row = cur.fetchone()
                return dict(row) if row else None

    def upsert(
        self,
        network: str,
        identity_hash: str,
        *,
        username: str | None = None,
        beneficiary_address: str | None = None,
        vault_object_id: str | None = None,
        provision_tx_digest: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO username_beneficiaries (
                        network, identity_hash, username, beneficiary_address,
                        vault_object_id, provision_tx_digest, metadata, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                    ON CONFLICT (network, identity_hash) DO UPDATE SET
                        username = COALESCE(EXCLUDED.username, username_beneficiaries.username),
                        beneficiary_address = COALESCE(EXCLUDED.beneficiary_address, username_beneficiaries.beneficiary_address),
                        vault_object_id = COALESCE(EXCLUDED.vault_object_id, username_beneficiaries.vault_object_id),
                        provision_tx_digest = COALESCE(EXCLUDED.provision_tx_digest, username_beneficiaries.provision_tx_digest),
                        metadata = COALESCE(EXCLUDED.metadata, username_beneficiaries.metadata),
                        updated_at = NOW()
                    """,
                    (
                        network,
                        identity_hash,
                        username,
                        beneficiary_address,
                        vault_object_id,
                        provision_tx_digest,
                        json.dumps(metadata or {}),
                    ),
                )
                conn.commit()

    def mark_claimed(self, network: str, identity_hash: str, claim_tx_digest: str) -> None:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE username_beneficiaries
                    SET claimed = 'true', claim_tx_digest = %s, updated_at = NOW()
                    WHERE network = %s AND identity_hash = %s
                    """,
                    (claim_tx_digest, network, identity_hash),
                )
                conn.commit()


class MediaAssetRepository:
    def get(self, network: str, asset_id: str) -> dict | None:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM chain_media_assets WHERE network = %s AND asset_id = %s",
                    (network, asset_id),
                )
                row = cur.fetchone()
                return dict(row) if row else None

    def upsert(self, network: str, record: dict) -> None:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chain_media_assets (
                        network, asset_id, content_commitment, fingerprint_commitment,
                        media_type, originality_status, lineage_parent_id, creators,
                        rights_controllers, beneficiaries, beneficiary_splits, rights_json,
                        rights_version, economics_version, linked_existing, resolve_tx_digest,
                        request_id, metadata, registered_at, updated_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW()
                    )
                    ON CONFLICT (network, asset_id) DO UPDATE SET
                        fingerprint_commitment = COALESCE(EXCLUDED.fingerprint_commitment, chain_media_assets.fingerprint_commitment),
                        resolve_tx_digest = COALESCE(EXCLUDED.resolve_tx_digest, chain_media_assets.resolve_tx_digest),
                        metadata = COALESCE(EXCLUDED.metadata, chain_media_assets.metadata),
                        updated_at = NOW()
                    """,
                    (
                        network,
                        record["asset_id"],
                        record.get("content_commitment"),
                        record.get("fingerprint_commitment"),
                        record.get("media_type"),
                        record.get("originality_status"),
                        record.get("lineage_parent_id"),
                        json.dumps(record.get("creators") or []),
                        json.dumps(record.get("rights_controllers") or []),
                        json.dumps(record.get("beneficiaries") or []),
                        json.dumps(record.get("beneficiary_splits") or []),
                        json.dumps(record.get("rights_json") or {}),
                        record.get("rights_version", 1),
                        record.get("economics_version", 1),
                        bool(record.get("linked_existing")),
                        record.get("resolve_tx_digest"),
                        record.get("request_id"),
                        json.dumps(record.get("metadata") or {}),
                    ),
                )
                conn.commit()

    def find_by_fingerprint(self, network: str, fingerprint_commitment: str) -> dict | None:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM chain_media_assets
                    WHERE network = %s AND fingerprint_commitment = %s
                    ORDER BY registered_at DESC
                    LIMIT 1
                    """,
                    (network, fingerprint_commitment),
                )
                row = cur.fetchone()
                return dict(row) if row else None


class FingerprintObservationRepository:
    def record(self, network: str, record: dict) -> str:
        obs_id = str(uuid.uuid4())
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO fingerprint_observations (
                        id, network, fingerprint_commitment, content_commitment,
                        media_asset_id, request_id, media_type, submitter, observed_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    """,
                    (
                        obs_id,
                        network,
                        record["fingerprint_commitment"],
                        record.get("content_commitment"),
                        record.get("media_asset_id"),
                        record.get("request_id"),
                        record.get("media_type"),
                        record.get("submitter"),
                    ),
                )
                conn.commit()
        return obs_id

    def find_asset_for_fingerprint(self, network: str, fingerprint_commitment: str) -> str | None:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT media_asset_id FROM fingerprint_observations
                    WHERE network = %s AND fingerprint_commitment = %s AND media_asset_id IS NOT NULL
                    ORDER BY observed_at DESC
                    LIMIT 1
                    """,
                    (network, fingerprint_commitment),
                )
                row = cur.fetchone()
                return str(row[0]) if row and row[0] else None


class MediaAssetUsageRepository:
    def list_for_asset(self, network: str, asset_id: str, *, limit: int = 50) -> list[dict]:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM media_asset_usages
                    WHERE network = %s AND asset_id = %s
                    ORDER BY observed_at DESC
                    LIMIT %s
                    """,
                    (network, asset_id, limit),
                )
                return [dict(r) for r in cur.fetchall()]

    def record(self, network: str, record: dict) -> str:
        usage_id = str(uuid.uuid4())
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO media_asset_usages (
                        id, network, asset_id, container_id, container_type,
                        usage_class, position, tx_digest, observed_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    """,
                    (
                        usage_id,
                        network,
                        record["asset_id"],
                        record["container_id"],
                        int(record.get("container_type") or 1),
                        int(record.get("usage_class") or 1),
                        int(record.get("position") or 0),
                        record.get("tx_digest"),
                    ),
                )
                conn.commit()
        return usage_id


class CompositionAnalysisRepository:
    def insert(self, network: str, record: dict) -> str:
        rec_id = str(uuid.uuid4())
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO composition_analysis_records (
                        id, network, post_id, composition_status, monetization_status,
                        analysis_json, manifest_json, reasoning, evidence_urls,
                        contains_derivatives, contains_unresolved_assets, tx_digest, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    """,
                    (
                        rec_id,
                        network,
                        record["post_id"],
                        record.get("composition_status"),
                        record.get("monetization_status"),
                        json.dumps(record.get("analysis_json") or {}),
                        json.dumps(record.get("manifest_json")) if record.get("manifest_json") is not None else None,
                        record.get("reasoning"),
                        json.dumps(record.get("evidence_urls") or []),
                        bool(record.get("contains_derivatives")),
                        bool(record.get("contains_unresolved_assets")),
                        record.get("tx_digest"),
                    ),
                )
                conn.commit()
        return rec_id

    def latest_for_post(self, network: str, post_id: str) -> dict | None:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM composition_analysis_records
                    WHERE network = %s AND post_id = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (network, post_id),
                )
                row = cur.fetchone()
                return dict(row) if row else None


class ExternalIdentityRepository:
    def upsert(
        self,
        network: str,
        identity_hash: str,
        *,
        platform: str | None = None,
        external_id: str | None = None,
        display_name: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO external_identities (
                        network, identity_hash, platform, external_id, display_name, metadata, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (network, identity_hash) DO UPDATE SET
                        platform = COALESCE(EXCLUDED.platform, external_identities.platform),
                        external_id = COALESCE(EXCLUDED.external_id, external_identities.external_id),
                        display_name = COALESCE(EXCLUDED.display_name, external_identities.display_name),
                        metadata = COALESCE(EXCLUDED.metadata, external_identities.metadata)
                    """,
                    (
                        network,
                        identity_hash,
                        platform,
                        external_id,
                        json.dumps(metadata or {}),
                    ),
                )
                conn.commit()


class PendingDerivativeAssetRepository:
    def upsert(self, network: str, record: dict) -> None:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO pending_derivative_assets (
                        network, pending_id, request_id, content_commitment,
                        fingerprint_commitment, media_type, asset_kind, creator, status,
                        finalize_tx_digest, child_asset_id, metadata, created_at, updated_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW()
                    )
                    ON CONFLICT (network, pending_id) DO UPDATE SET
                        status = COALESCE(EXCLUDED.status, pending_derivative_assets.status),
                        finalize_tx_digest = COALESCE(EXCLUDED.finalize_tx_digest, pending_derivative_assets.finalize_tx_digest),
                        child_asset_id = COALESCE(EXCLUDED.child_asset_id, pending_derivative_assets.child_asset_id),
                        metadata = COALESCE(EXCLUDED.metadata, pending_derivative_assets.metadata),
                        updated_at = NOW()
                    """,
                    (
                        network,
                        record["pending_id"],
                        record.get("request_id"),
                        record.get("content_commitment"),
                        record.get("fingerprint_commitment"),
                        record.get("media_type"),
                        record.get("asset_kind", 0),
                        record.get("creator"),
                        record.get("status", "pending"),
                        record.get("finalize_tx_digest"),
                        record.get("child_asset_id"),
                        json.dumps(record.get("metadata") or {}),
                    ),
                )
                conn.commit()

    def get(self, network: str, pending_id: str) -> dict | None:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM pending_derivative_assets WHERE network = %s AND pending_id = %s",
                    (network, pending_id),
                )
                row = cur.fetchone()
                return dict(row) if row else None

    def find_by_request(self, network: str, request_id: str) -> dict | None:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM pending_derivative_assets
                    WHERE network = %s AND request_id = %s
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (network, request_id),
                )
                row = cur.fetchone()
                return dict(row) if row else None


class DerivativeEdgeRepository:
    def record(self, network: str, record: dict) -> str:
        edge_id = str(uuid.uuid4())
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO derivative_edges (
                        id, network, parent_asset_id, child_asset_id, relationship_type,
                        license_instance_id, template_version_id, parent_share_bps, tx_digest, observed_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    """,
                    (
                        edge_id,
                        network,
                        record["parent_asset_id"],
                        record["child_asset_id"],
                        record.get("relationship_type", 1),
                        record.get("license_instance_id"),
                        record.get("template_version_id"),
                        record.get("parent_share_bps"),
                        record.get("tx_digest"),
                    ),
                )
                conn.commit()
        return edge_id

    def list_for_asset(self, network: str, asset_id: str) -> list[dict]:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM derivative_edges
                    WHERE network = %s AND (child_asset_id = %s OR parent_asset_id = %s)
                    ORDER BY observed_at DESC
                    """,
                    (network, asset_id, asset_id),
                )
                return [dict(r) for r in cur.fetchall()]


class DetectedRelationshipRepository:
    def upsert(self, network: str, record: dict) -> None:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO detected_asset_relationships (
                        network, proposal_id, accused_pending_id, accused_asset_id,
                        original_asset_id, similarity_bps, status, evidence_commitment,
                        tx_digest, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                    ON CONFLICT (network, proposal_id) DO UPDATE SET
                        accused_asset_id = COALESCE(EXCLUDED.accused_asset_id, detected_asset_relationships.accused_asset_id),
                        status = EXCLUDED.status,
                        tx_digest = COALESCE(EXCLUDED.tx_digest, detected_asset_relationships.tx_digest),
                        updated_at = NOW()
                    """,
                    (
                        network,
                        record["proposal_id"],
                        record["accused_pending_id"],
                        record.get("accused_asset_id"),
                        record["original_asset_id"],
                        int(record["similarity_bps"]),
                        int(record.get("status", 0)),
                        record.get("evidence_commitment"),
                        record.get("tx_digest"),
                    ),
                )
                conn.commit()

    def list_for_pending(self, network: str, pending_id: str) -> list[dict]:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM detected_asset_relationships
                    WHERE network = %s AND accused_pending_id = %s
                    ORDER BY created_at DESC
                    """,
                    (network, pending_id),
                )
                return [dict(r) for r in cur.fetchall()]


class PostEnforcementSnapshotRepository:
    def upsert(self, network: str, record: dict) -> None:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO post_enforcement_snapshots (
                        network, post_id, bindings_json, usage_decisions_json,
                        usage_denials_json, playback_policy_json, tx_digest, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (network, post_id) DO UPDATE SET
                        bindings_json = EXCLUDED.bindings_json,
                        usage_decisions_json = EXCLUDED.usage_decisions_json,
                        usage_denials_json = EXCLUDED.usage_denials_json,
                        playback_policy_json = EXCLUDED.playback_policy_json,
                        tx_digest = COALESCE(EXCLUDED.tx_digest, post_enforcement_snapshots.tx_digest),
                        updated_at = NOW()
                    """,
                    (
                        network,
                        record["post_id"],
                        json.dumps(record.get("bindings_json") or []),
                        json.dumps(record.get("usage_decisions_json") or []),
                        json.dumps(record.get("usage_denials_json") or []),
                        json.dumps(record.get("playback_policy_json") or {}),
                        record.get("tx_digest"),
                    ),
                )
                conn.commit()

    def get(self, network: str, post_id: str) -> dict | None:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM post_enforcement_snapshots WHERE network = %s AND post_id = %s",
                    (network, post_id),
                )
                row = cur.fetchone()
                return dict(row) if row else None


class MediaAssetRightsBundleRepository:
    def upsert(
        self,
        network: str,
        *,
        proposal_id: str,
        media_asset_id: str,
        claims_commitment: bytes,
        claims_bcs: bytes,
        usage_grants_bcs: bytes,
        submitter: str,
        status: str = "pending",
    ) -> None:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO media_asset_rights_bundles (
                        proposal_id, network, media_asset_id, claims_commitment,
                        claims_bcs, usage_grants_bcs, submitter, status
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (proposal_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        claims_commitment = EXCLUDED.claims_commitment
                    """,
                    (
                        proposal_id,
                        network,
                        media_asset_id,
                        claims_commitment,
                        claims_bcs,
                        usage_grants_bcs,
                        submitter,
                        status,
                    ),
                )
                conn.commit()

    def get(self, proposal_id: str) -> dict[str, Any] | None:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT proposal_id, network, media_asset_id, claims_commitment,
                           claims_bcs, usage_grants_bcs, submitter, status, created_at
                    FROM media_asset_rights_bundles
                    WHERE proposal_id = %s
                    """,
                    (proposal_id,),
                )
                row = cur.fetchone()
                return dict(row) if row else None

    def mark_status(self, proposal_id: str, status: str) -> None:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE media_asset_rights_bundles
                    SET status = %s
                    WHERE proposal_id = %s
                    """,
                    (status, proposal_id),
                )
                conn.commit()
