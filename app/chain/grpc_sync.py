"""gRPC blockchain sync — ingests PostCreatedEvent and enqueues oracle jobs."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import structlog

from app.chain.chain_types import CHECKPOINT_MARKER_EVENT
from app.chain.event_parser import infer_media_type, parse_post_created
from app.chain.grpc_client import (
    GrpcChainClient,
    resolve_checkpoint_stream_id,
    resolve_event_stream_id,
    resolve_sync_mode,
    resolve_sync_start_checkpoint,
)
from app.chain.grpc_v2_client import GrpcV2Client
from app.db.oracle_repository import ChainPostRepository, CheckpointRepository, JobRepository
from app.network_config import load_network_profile
from app.services.events import event_bus

logger = structlog.get_logger()


class GrpcSyncService:
    def __init__(self, network: str) -> None:
        self.network = network
        self.profile = load_network_profile(network)
        self.stream_id = resolve_event_stream_id(self.profile)
        self.checkpoint_stream_id = resolve_checkpoint_stream_id(self.profile)
        self.sync_mode = resolve_sync_mode(self.profile)
        self.client = GrpcChainClient(self.profile)
        self.posts = ChainPostRepository()
        self.checkpoints = CheckpointRepository()
        self.jobs = JobRepository()
        self.last_event_at: datetime | None = None

    def _compute_lag(self) -> int:
        highest = self.client.last_highest_indexed_checkpoint
        cp = self.checkpoints.get(self.network, self.checkpoint_stream_id) or {}
        last = int(cp.get("checkpoint_sequence") or 0)
        if highest <= 0:
            return 0
        return max(0, highest - last)

    async def _save_checkpoint(self, checkpoint_sequence: int, transaction_digest: str | None) -> None:
        self.checkpoints.save(
            self.network,
            checkpoint_sequence,
            stream_id=self.checkpoint_stream_id,
            last_transaction_digest=transaction_digest,
        )
        self.last_event_at = datetime.now(timezone.utc)
        lag = self._compute_lag()
        await event_bus.publish(
            "sync.checkpoint",
            {
                "network": self.network,
                "stream_id": self.checkpoint_stream_id,
                "package_id": self.stream_id,
                "sync_mode": self.sync_mode,
                "checkpoint": checkpoint_sequence,
                "lag": lag,
                "highest_indexed_checkpoint": self.client.last_highest_indexed_checkpoint,
                "mock_mode": self.profile.grpc_sync.mock_mode,
            },
        )

    async def _resolve_start_checkpoint(self) -> int:
        cp = self.checkpoints.get(self.network, self.checkpoint_stream_id) or {}
        saved_raw = cp.get("checkpoint_sequence")
        saved = int(saved_raw) if saved_raw is not None else None
        start = resolve_sync_start_checkpoint(
            saved_checkpoint_sequence=saved,
            configured_start=self.profile.grpc_sync.start_checkpoint,
        )
        if self.sync_mode == "mock":
            return start
        try:
            v2 = GrpcV2Client(self.profile)
            try:
                tip = v2.get_chain_tip()
                self.client.last_highest_indexed_checkpoint = tip
                if start > tip:
                    logger.warning(
                        "Saved checkpoint ahead of chain tip — resetting sync cursor",
                        network=self.network,
                        saved_checkpoint=start,
                        chain_tip=tip,
                    )
                    self.checkpoints.save(
                        self.network,
                        0,
                        stream_id=self.checkpoint_stream_id,
                    )
                    return 0
            finally:
                v2.close()
        except Exception as exc:
            logger.warning(
                "Could not verify chain tip for checkpoint reset",
                network=self.network,
                error=str(exc),
            )
        return start

    async def run_forever(self) -> None:
        start = await self._resolve_start_checkpoint()
        logger.info(
            "Starting gRPC sync",
            network=self.network,
            stream_id=self.checkpoint_stream_id,
            package_id=self.stream_id,
            sync_mode=self.sync_mode,
            start_checkpoint=start,
            mock_mode=self.profile.grpc_sync.mock_mode,
        )
        while True:
            try:
                async for event in self.client.stream_events(start_checkpoint=start):
                    if event.event_type == CHECKPOINT_MARKER_EVENT:
                        await self._save_checkpoint(event.checkpoint_sequence, None)
                        start = event.checkpoint_sequence
                        continue
                    await self._handle_event(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("gRPC sync error", network=self.network, error=str(exc))
                await event_bus.publish(
                    "sync.error",
                    {
                        "network": self.network,
                        "stream_id": self.checkpoint_stream_id,
                        "package_id": self.stream_id,
                        "error": str(exc),
                    },
                )
                await asyncio.sleep(self.profile.grpc_sync.poll_interval_seconds)

    async def _handle_event(self, event) -> None:
        if event.event_type != "PostCreatedEvent":
            return
        parsed = parse_post_created(event.payload)
        if not parsed:
            return
        if not parsed["enable_poc"] or not parsed["media_urls"]:
            return

        inserted = self.posts.upsert_discovered(
            self.network,
            parsed["post_id"],
            creator_address=parsed.get("creator"),
            enable_poc=True,
            media_urls=parsed["media_urls"],
            media_types=parsed.get("media_types") or [],
            metadata={"tx_digest": event.transaction_digest},
        )
        if not inserted:
            return

        await event_bus.publish(
            "post.discovered",
            {
                "network": self.network,
                "post_id": parsed["post_id"],
                "media_urls": parsed["media_urls"],
                "enable_poc": True,
            },
        )

        media_types = parsed.get("media_types") or []
        for idx, url in enumerate(parsed["media_urls"]):
            media_type = infer_media_type(url, idx, media_types)
            job_id = self.jobs.enqueue(
                self.network,
                parsed["post_id"],
                job_type="analyze_post",
                media_url=url,
                media_index=idx,
                media_type=media_type,
            )
            logger.info(
                "Enqueued analyze job",
                network=self.network,
                post_id=parsed["post_id"],
                job_id=job_id,
                media_url=url,
            )


async def run_sync_for_network(network: str) -> None:
    service = GrpcSyncService(network)
    await service.run_forever()
