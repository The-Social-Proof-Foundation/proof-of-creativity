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
        metadata: dict | None = None,
    ) -> bool:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chain_posts (
                        network, post_id, creator_address, enable_poc, media_urls,
                        media_types, analysis_status, metadata, discovered_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, 'discovered', %s, NOW(), NOW())
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
                        display_name,
                        json.dumps(metadata or {}),
                    ),
                )
                conn.commit()
