"""Discovery worker — processes embed jobs internally (no HTTP round-trip)."""

from __future__ import annotations

import asyncio

import structlog
from fastapi import HTTPException

from app.discovery.embedding_service import embed_discovered_asset
from app.discovery.lifecycle import LifecycleEvent
from app.discovery.store import DiscoveryStore
from app.discovery.scheduler import DiscoveryScheduler, poll_interval_seconds

logger = structlog.get_logger()


class DiscoveryWorker:
    def __init__(self) -> None:
        self.repo = DiscoveryStore()
        self.scheduler = DiscoveryScheduler()

    async def run_forever(self) -> None:
        logger.info("Discovery worker started", poll_interval=poll_interval_seconds())
        poll_counter = 0
        while True:
            if poll_counter <= 0:
                try:
                    stats = self.scheduler.poll_once()
                    if stats["sources"]:
                        logger.info("Discovery scheduler poll complete", **stats)
                except Exception as exc:
                    logger.exception("Discovery scheduler poll failed", error=str(exc))
                poll_counter = max(1, poll_interval_seconds())
            poll_counter -= 1

            job = self.repo.claim_next_embed_job()
            if not job:
                await asyncio.sleep(1.0)
                continue
            await self._process_job(job)

    async def _process_job(self, job: dict) -> None:
        job_id = str(job["id"])
        asset_id = str(job["discovery_asset_id"])
        try:
            asset = self.repo.asset_for_embed(asset_id)
            if not asset:
                raise ValueError(f"discovery asset not found: {asset_id}")

            state = self.repo.asset_lifecycle_state(asset_id)
            if state and state.is_at_least_indexed():
                self.repo.complete_job(job_id, status="completed")
                return

            self.repo.transition_asset(asset_id, LifecycleEvent.START_ACQUIRE)
            result = await embed_discovered_asset(
                discovery_asset_id=asset_id,
                external_source_url=asset["external_source_url"],
                media_type=asset["media_type"],
                creator_x_handle=asset.get("creator_x_handle"),
                creator_confidence=float(asset.get("creator_confidence") or 0),
                creator_candidate_id=asset.get("creator_candidate_id"),
            )
            self.repo.transition_asset(asset_id, LifecycleEvent.EMBED_COMPLETE)
            self.repo.transition_asset(asset_id, LifecycleEvent.INDEX_COMPLETE)
            self.repo.update_embed_result(
                asset_id,
                work_confidence=result.work_confidence,
                embedding_version=result.embedding_version,
                identity_hash=result.identity_hash,
            )
            self.repo.complete_job(job_id, status="completed")
            logger.info(
                "Discovery embed job completed",
                job_id=job_id,
                asset_id=asset_id,
                media_id=result.media_id,
                work_confidence=result.work_confidence,
            )
        except HTTPException as exc:
            logger.warning(
                "Discovery embed job failed (client error)",
                job_id=job_id,
                asset_id=asset_id,
                status=exc.status_code,
                detail=exc.detail,
            )
            self.repo.transition_asset(asset_id, LifecycleEvent.FAIL)
            attempts = int(job.get("attempts") or 1)
            max_attempts = int(job.get("max_attempts") or 5)
            if attempts >= max_attempts:
                self.repo.complete_job(job_id, status="failed", error=str(exc.detail))
            else:
                self.repo.defer_job(job_id, delay_seconds=30, error=str(exc.detail))
        except Exception as exc:
            logger.exception("Discovery embed job failed", job_id=job_id, asset_id=asset_id, error=str(exc))
            try:
                self.repo.transition_asset(asset_id, LifecycleEvent.FAIL)
            except Exception:
                pass
            attempts = int(job.get("attempts") or 1)
            max_attempts = int(job.get("max_attempts") or 5)
            if attempts >= max_attempts:
                self.repo.complete_job(job_id, status="failed", error=str(exc))
            else:
                self.repo.defer_job(job_id, delay_seconds=30, error=str(exc))


async def run_discovery_worker() -> None:
    await DiscoveryWorker().run_forever()
