"""Startup readiness for off-network discovery tables."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.database import get_db_connection
from app.discovery.store import DiscoveryStore


@dataclass
class DiscoveryBootstrapStatus:
    ready: bool
    migration_applied: bool = False
    asset_counts: dict[str, int] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)


def _migration_applied() -> bool:
    sql = """
    SELECT 1 FROM information_schema.tables
    WHERE table_schema = 'public' AND table_name = 'discovery_assets'
    LIMIT 1
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchone() is not None


def evaluate_bootstrap_status() -> DiscoveryBootstrapStatus:
    issues: list[str] = []
    migration = _migration_applied()
    if not migration:
        issues.append("discovery_assets table missing — run alembic upgrade head (f3a4b5c6d7e8+)")
    counts: dict[str, int] = {}
    if migration:
        try:
            counts = DiscoveryStore().asset_counts()
        except Exception as exc:
            issues.append(f"discovery stats query failed: {exc}")
    ready = migration and not issues
    return DiscoveryBootstrapStatus(
        ready=ready,
        migration_applied=migration,
        asset_counts=counts,
        issues=issues,
    )
