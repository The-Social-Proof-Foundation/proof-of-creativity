#!/usr/bin/env python3
"""Re-embed discovered corpus rows at POC_ACTIVE_EMBEDDING_VERSION."""

from __future__ import annotations

import asyncio
import os
import sys

import structlog

from app.core.database import get_db_connection
from app.discovery.context import default_embedding_version
from app.discovery.embedding_service import embed_discovered_asset
from app.discovery.store import DiscoveryStore

logger = structlog.get_logger()


def stale_discovered_assets(limit: int = 100) -> list[dict]:
    active = os.getenv("POC_ACTIVE_EMBEDDING_VERSION", default_embedding_version())
    sql = """
    SELECT DISTINCT discovery_asset_id::text AS discovery_asset_id,
           metadata->>'external_source_url' AS external_source_url,
           metadata->>'media_type' AS media_type,
           metadata->>'creator_x_handle' AS creator_x_handle,
           COALESCE((metadata->>'creator_confidence')::float, 0) AS creator_confidence,
           metadata->>'creator_candidate_id' AS creator_candidate_id
    FROM media_embeddings
    WHERE corpus_scope = 'discovered'
      AND embedding_version IS DISTINCT FROM %s
      AND discovery_asset_id IS NOT NULL
    LIMIT %s
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (active, limit))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


async def reembed_one(row: dict) -> bool:
    asset_id = row["discovery_asset_id"]
    url = row.get("external_source_url")
    media_type = row.get("media_type") or "image"
    if not url:
        asset = DiscoveryStore().asset_for_embed(asset_id)
        if asset:
            url = asset.get("external_source_url")
            media_type = asset.get("media_type") or media_type
    if not url:
        logger.warning("Skipping re-embed without source URL", asset_id=asset_id)
        return False
    await embed_discovered_asset(
        discovery_asset_id=asset_id,
        external_source_url=url,
        media_type=media_type,
        creator_x_handle=row.get("creator_x_handle"),
        creator_confidence=float(row.get("creator_confidence") or 0),
        creator_candidate_id=row.get("creator_candidate_id"),
    )
    return True


async def main() -> int:
    rows = stale_discovered_assets()
    if not rows:
        logger.info("No stale discovered embeddings")
        return 0
    ok = 0
    for row in rows:
        try:
            if await reembed_one(row):
                ok += 1
        except Exception as exc:
            logger.warning("Re-embed failed", asset_id=row.get("discovery_asset_id"), error=str(exc))
    logger.info("Re-embed complete", processed=ok, total=len(rows))
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
