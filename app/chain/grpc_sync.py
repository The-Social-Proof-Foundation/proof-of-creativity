"""gRPC blockchain sync — ingests chain events and enqueues oracle jobs."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import structlog

from app.chain.chain_types import CHECKPOINT_MARKER_EVENT
from app.chain.event_parser import (
    infer_media_type,
    infer_primary_media_type,
    parse_post_created,
    parse_post_metadata_commitments,
)
from app.chain.grpc_client import (
    GrpcChainClient,
    resolve_checkpoint_stream_id,
    resolve_event_stream_id,
    resolve_sync_mode,
    resolve_sync_start_checkpoint,
)
from app.chain.grpc_v2_client import GrpcV2Client
from app.db.oracle_repository import (
    ChainPostRepository,
    CheckpointRepository,
    DetectedRelationshipRepository,
    DerivativeEdgeRepository,
    JobRepository,
    MediaAssetRepository,
    PendingDerivativeAssetRepository,
    PostEnforcementSnapshotRepository,
)
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
        self.media_assets = MediaAssetRepository()
        self.pending_assets = PendingDerivativeAssetRepository()
        self.derivative_edges = DerivativeEdgeRepository()
        self.detected_relationships = DetectedRelationshipRepository()
        self.post_enforcement = PostEnforcementSnapshotRepository()

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
        if event.event_type == "MediaResolutionRequestedEvent":
            await self._handle_media_resolution_requested(event)
            return
        if event.event_type == "DerivativeAssetFinalizedEvent":
            await self._handle_derivative_asset_finalized(event)
            return
        if event.event_type == "OriginalAssetFinalizedEvent":
            await self._handle_original_asset_finalized(event)
            return
        if event.event_type in {
            "ResolvedPolicyUpdatedEvent",
            "EmbeddedBindingRecordedEvent",
            "UsageDecisionRefreshedEvent",
            "ContainerUsageDeniedEvent",
            "RelationshipFinalizedEvent",
        }:
            await self._handle_enforcement_or_policy_event(event)
            return
        if event.event_type != "PostCreatedEvent":
            return
        await self._handle_post_created(event)

    async def _handle_derivative_asset_finalized(self, event) -> None:
        payload = dict(event.payload or {})
        pending_id = str(payload.get("pending_id") or "")
        child_asset_id = str(payload.get("child_asset_id") or "")
        if not child_asset_id:
            return
        if pending_id:
            self.pending_assets.upsert(
                self.network,
                {
                    "pending_id": pending_id,
                    "status": "finalized",
                    "child_asset_id": child_asset_id,
                    "finalize_tx_digest": event.transaction_digest,
                },
            )
        self.media_assets.upsert(
            self.network,
            {
                "asset_id": child_asset_id,
                "pending_id": pending_id or None,
                "originality_status": 2,
                "resolve_tx_digest": event.transaction_digest,
            },
        )
        for edge in payload.get("finalized_edges") or []:
            if not isinstance(edge, dict):
                continue
            parent_id = edge.get("parent_asset_id")
            if parent_id:
                self.derivative_edges.record(
                    self.network,
                    {
                        "parent_asset_id": str(parent_id),
                        "child_asset_id": child_asset_id,
                        "relationship_type": int(edge.get("relationship_type") or 1),
                        "license_instance_id": edge.get("license_instance_id"),
                        "template_version_id": edge.get("template_version_id"),
                        "parent_share_bps": edge.get("parent_share_bps"),
                        "tx_digest": event.transaction_digest,
                    },
                )

    async def _handle_original_asset_finalized(self, event) -> None:
        payload = dict(event.payload or {})
        pending_id = str(payload.get("pending_id") or "")
        child_asset_id = str(payload.get("child_asset_id") or "")
        if not child_asset_id:
            return
        if pending_id:
            self.pending_assets.upsert(
                self.network,
                {
                    "pending_id": pending_id,
                    "status": "finalized",
                    "child_asset_id": child_asset_id,
                    "finalize_tx_digest": event.transaction_digest,
                },
            )
        self.media_assets.upsert(
            self.network,
            {
                "asset_id": child_asset_id,
                "pending_id": pending_id or None,
                "originality_status": 1,
                "resolve_tx_digest": event.transaction_digest,
            },
        )
        self.jobs.enqueue(
            self.network,
            child_asset_id,
            job_type="materialize_policy",
            payload={"asset_id": child_asset_id},
        )

    async def _handle_enforcement_or_policy_event(self, event) -> None:
        payload = dict(event.payload or {})
        post_id = str(payload.get("post_id") or payload.get("container_id") or "")
        if event.event_type == "RelationshipFinalizedEvent":
            proposal_id = str(payload.get("proposal_id") or "")
            if proposal_id:
                self.detected_relationships.upsert(
                    self.network,
                    {
                        "proposal_id": proposal_id,
                        "accused_pending_id": str(payload.get("accused_pending_id") or ""),
                        "accused_asset_id": payload.get("accused_asset_id"),
                        "original_asset_id": str(payload.get("original_asset_id") or ""),
                        "similarity_bps": int(payload.get("similarity_bps") or 0),
                        "status": 3,
                        "tx_digest": event.transaction_digest,
                    },
                )
            return
        if not post_id:
            return
        snapshot = self.post_enforcement.get(self.network, post_id) or {
            "post_id": post_id,
            "bindings_json": [],
            "usage_decisions_json": [],
            "usage_denials_json": [],
            "playback_policy_json": {},
        }
        if event.event_type == "EmbeddedBindingRecordedEvent":
            bindings = list(snapshot.get("bindings_json") or [])
            bindings.append(payload)
            snapshot["bindings_json"] = bindings
        elif event.event_type == "UsageDecisionRefreshedEvent":
            decisions = list(snapshot.get("usage_decisions_json") or [])
            decisions.append(payload)
            snapshot["usage_decisions_json"] = decisions
        elif event.event_type == "ContainerUsageDeniedEvent":
            denials = list(snapshot.get("usage_denials_json") or [])
            denials.append(payload)
            snapshot["usage_denials_json"] = denials
        snapshot["tx_digest"] = event.transaction_digest
        self.post_enforcement.upsert(self.network, snapshot)

    async def _handle_media_resolution_requested(self, event) -> None:
        payload = dict(event.payload or {})
        request_id = payload.get("request_id")
        if not request_id:
            return
        job_id = self.jobs.enqueue(
            self.network,
            post_id=str(request_id),
            job_type="resolve_media_asset",
            payload={
                **payload,
                "tx_digest": event.transaction_digest,
                "event_sequence": getattr(event, "event_idx", 0) or 0,
            },
        )
        logger.info(
            "Enqueued resolve_media_asset job",
            network=self.network,
            request_id=request_id,
            job_id=job_id,
        )

    async def _handle_post_created(self, event) -> None:
        parsed = parse_post_created(event.payload)
        if not parsed:
            return

        media_asset_ids = parsed.get("media_asset_ids") or []
        media_urls = parsed.get("media_urls") or []
        enable_poc = parsed.get("enable_poc", True)

        if not media_asset_ids and (not enable_poc or not media_urls):
            return

        inserted = self.posts.upsert_discovered(
            self.network,
            parsed["post_id"],
            creator_address=parsed.get("creator"),
            enable_poc=bool(enable_poc),
            media_urls=media_urls,
            media_types=parsed.get("media_types") or [],
            media_asset_ids=media_asset_ids,
            composition_status=parsed.get("composition_status"),
            monetization_status=parsed.get("monetization_status"),
            metadata={
                "tx_digest": event.transaction_digest,
                "event_sequence": getattr(event, "event_idx", 0) or 0,
                "spt_id": parsed.get("spt_id"),
            },
        )
        if not inserted:
            return

        await event_bus.publish(
            "post.discovered",
            {
                "network": self.network,
                "post_id": parsed["post_id"],
                "media_urls": media_urls,
                "media_asset_ids": media_asset_ids,
                "enable_poc": bool(enable_poc),
            },
        )

        if media_asset_ids:
            job_id = self.jobs.enqueue(
                self.network,
                parsed["post_id"],
                job_type="analyze_composition",
                payload={
                    "media_asset_ids": media_asset_ids,
                    "spt_id": parsed.get("spt_id"),
                    "tx_digest": event.transaction_digest,
                },
            )
            logger.info(
                "Enqueued analyze_composition job",
                network=self.network,
                post_id=parsed["post_id"],
                job_id=job_id,
                asset_count=len(media_asset_ids),
            )
            return

        metadata_json = parsed.get("metadata_json") or (event.payload or {}).get("metadata_json")
        commitments = parse_post_metadata_commitments(metadata_json)
        if commitments and media_urls:
            media_types = parsed.get("media_types") or []
            job_id = self.jobs.enqueue(
                self.network,
                parsed["post_id"],
                job_type="resolve_post_media",
                payload={
                    "post_id": parsed["post_id"],
                    "content_commitment": commitments["content_commitment"],
                    "observed_fingerprint_commitment": commitments["observed_fingerprint_commitment"],
                    "media_type": infer_primary_media_type(media_urls, media_types),
                    "submitter": parsed.get("creator") or parsed.get("owner") or "",
                    "media_urls": media_urls,
                    "storage_asset_id": commitments.get("asset_id") or "",
                    "spt_id": parsed.get("spt_id"),
                    "tx_digest": event.transaction_digest,
                },
            )
            logger.info(
                "Enqueued resolve_post_media job",
                network=self.network,
                post_id=parsed["post_id"],
                job_id=job_id,
            )
            return

        media_types = parsed.get("media_types") or []
        for idx, url in enumerate(media_urls):
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
