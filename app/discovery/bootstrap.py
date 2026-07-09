"""Discovery corpus bootstrap and readiness checks."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog

from app.core.database import get_db_connection
from app.discovery.context import default_embedding_version

logger = structlog.get_logger()

REQUIRED_MIGRATION = "d1e2f3a4b5c6"


@dataclass
class DiscoveryBootstrapStatus:
    ready: bool
    migration_applied: bool
    active_embedding_version: str
    corpus_counts: dict[str, int] = field(default_factory=dict)
    discovery_service_healthy: bool | None = None
    issues: list[str] = field(default_factory=list)


def _migration_applied(revision: str) -> bool:
    sql = "SELECT version_num FROM alembic_version LIMIT 1"
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                row = cur.fetchone()
        if not row:
            return False
        current = str(row[0])
        return current >= revision
    except Exception as exc:
        logger.warning("Could not read alembic_version", error=str(exc))
        return False


def corpus_counts() -> dict[str, int]:
    version = os.getenv("DISCOVERY_ACTIVE_EMBEDDING_VERSION", default_embedding_version())
    counts: dict[str, int] = {}
    queries = {
        "media_embeddings_discovered": (
            "SELECT COUNT(*) FROM media_embeddings WHERE corpus_scope = 'discovered' "
            "AND embedding_version = %s",
            (version,),
        ),
        "audio_fingerprints_discovered": (
            "SELECT COUNT(*) FROM audio_fingerprints WHERE corpus_scope = 'discovered' "
            "AND embedding_version = %s",
            (version,),
        ),
        "image_hashes_discovered": (
            "SELECT COUNT(*) FROM image_hashes WHERE corpus_scope = 'discovered' "
            "AND embedding_version = %s",
            (version,),
        ),
        "provenance_hits": ("SELECT COUNT(*) FROM provenance_hits", ()),
    }
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for key, (sql, params) in queries.items():
                cur.execute(sql, params)
                row = cur.fetchone()
                counts[key] = int(row[0]) if row else 0
    return counts


def check_discovery_service_health() -> bool | None:
    if os.getenv("DISCOVERY_ENABLED", "").lower() not in ("1", "true", "yes"):
        return None
    base = os.getenv("DISCOVERY_SERVICE_URL", "").rstrip("/")
    if not base:
        return None
    try:
        resp = httpx.get(f"{base}/health", timeout=5.0)
        return resp.status_code == 200
    except Exception as exc:
        logger.warning("Discovery service health check failed", error=str(exc))
        return False


def evaluate_bootstrap_status() -> DiscoveryBootstrapStatus:
    issues: list[str] = []
    migration_ok = _migration_applied(REQUIRED_MIGRATION)
    if not migration_ok:
        issues.append(f"alembic revision {REQUIRED_MIGRATION} not applied")

    counts = corpus_counts()
    ds_health = check_discovery_service_health()
    if ds_health is False:
        issues.append("discovery-service health check failed")

    ready = migration_ok and (ds_health is not False)
    status = DiscoveryBootstrapStatus(
        ready=ready,
        migration_applied=migration_ok,
        active_embedding_version=os.getenv(
            "DISCOVERY_ACTIVE_EMBEDDING_VERSION", default_embedding_version()
        ),
        corpus_counts=counts,
        discovery_service_healthy=ds_health,
        issues=issues,
    )
    return status


def bootstrap_status_dict() -> dict[str, Any]:
    status = evaluate_bootstrap_status()
    return {
        "ready": status.ready,
        "migration_applied": status.migration_applied,
        "active_embedding_version": status.active_embedding_version,
        "corpus_counts": status.corpus_counts,
        "discovery_service_healthy": status.discovery_service_healthy,
        "issues": status.issues,
    }
