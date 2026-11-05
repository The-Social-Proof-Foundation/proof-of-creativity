"""Social indexer GraphQL sync — ingests posts with media into oracle pipeline."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any

import httpx
import structlog

from app.chain.event_parser import infer_media_type
from app.chain.graphql_session import resolve_graphql_url
from app.db.oracle_repository import ChainPostRepository, JobRepository
from app.network_config import load_network_profile
from app.services.events import event_bus

logger = structlog.get_logger()

_POSTS_QUERY = """
query PocIndexerPosts($limit: Int) {
  posts(limit: $limit) {
    postId
    owner
    mediaUrls
    pocId
    pocAnalyzedAt
    pocMediaType
  }
}
"""


def _parse_media_urls(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(u) for u in raw if u]
    if isinstance(raw, dict) and "value" in raw:
        return _parse_media_urls(raw["value"])
    return []


class IndexerSyncService:
    def __init__(self, network: str) -> None:
        self.network = network
        self.profile = load_network_profile(network)
        self.posts = ChainPostRepository()
        self.jobs = JobRepository()
        self.poll_interval = float(os.getenv("POC_INDEXER_POLL_INTERVAL_SECONDS", "5"))
        self.batch_limit = int(os.getenv("POC_INDEXER_BATCH_LIMIT", "50"))
        self.last_poll_at: datetime | None = None

    def _graphql_url(self) -> str:
        return resolve_graphql_url(self.profile)

    async def _fetch_posts(self) -> list[dict[str, Any]]:
        url = self._graphql_url()
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                url,
                json={"query": _POSTS_QUERY, "variables": {"limit": self.batch_limit}},
            )
            resp.raise_for_status()
            body = resp.json()
        if body.get("errors"):
            raise RuntimeError(f"GraphQL errors: {body['errors']}")
        data = body.get("data") or {}
        posts = data.get("posts") or []
        return [p for p in posts if isinstance(p, dict)]

    async def _ingest_post(self, post: dict[str, Any]) -> None:
        post_id = post.get("postId") or post.get("post_id")
        if not post_id:
            return
        if post.get("pocAnalyzedAt") or post.get("poc_analyzed_at"):
            return
        media_urls = _parse_media_urls(post.get("mediaUrls") or post.get("media_urls"))
        if not media_urls:
            return
        enable_poc = bool(post.get("pocId") or post.get("poc_id") or True)
        if not enable_poc:
            return

        inserted = self.posts.upsert_discovered(
            self.network,
            str(post_id),
            creator_address=post.get("owner"),
            enable_poc=True,
            media_urls=media_urls,
            media_types=[int(post["pocMediaType"])] if post.get("pocMediaType") else [],
            metadata={"source": "indexer_graphql"},
        )
        if not inserted:
            return

        await event_bus.publish(
            "post.discovered",
            {
                "network": self.network,
                "post_id": str(post_id),
                "media_urls": media_urls,
                "enable_poc": True,
                "source": "indexer",
            },
        )

        media_type_hint = post.get("pocMediaType")
        for idx, url in enumerate(media_urls):
            media_type = (
                int(media_type_hint)
                if media_type_hint is not None and idx == 0
                else infer_media_type(url, idx, [])
            )
            job_id = self.jobs.enqueue(
                self.network,
                str(post_id),
                job_type="analyze_post",
                media_url=url,
                media_index=idx,
                media_type=media_type,
            )
            logger.info(
                "Enqueued analyze job from indexer",
                network=self.network,
                post_id=post_id,
                job_id=job_id,
                media_url=url,
            )

    async def poll_once(self) -> int:
        posts = await self._fetch_posts()
        ingested = 0
        for post in posts:
            before = self.posts.get(self.network, str(post.get("postId") or ""))
            await self._ingest_post(post)
            after = self.posts.get(self.network, str(post.get("postId") or ""))
            if not before and after:
                ingested += 1
        self.last_poll_at = datetime.now(timezone.utc)
        return ingested

    async def run_forever(self) -> None:
        logger.info(
            "Indexer sync started",
            network=self.network,
            graphql_url=self._graphql_url(),
            poll_interval=self.poll_interval,
        )
        while True:
            try:
                count = await self.poll_once()
                if count:
                    logger.info("Indexer sync ingested posts", count=count)
            except Exception as exc:
                logger.warning("Indexer sync poll failed", error=str(exc))
            await asyncio.sleep(self.poll_interval)


async def run_indexer_sync_for_network(network: str) -> None:
    service = IndexerSyncService(network)
    await service.run_forever()
