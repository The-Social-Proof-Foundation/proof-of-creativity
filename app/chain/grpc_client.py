"""gRPC client wrapper for MySocial blockchain sync."""

from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path
from typing import AsyncIterator

import grpc
import structlog

from app.chain.authenticated_events_client import AuthenticatedEventsClient
from app.chain.bcs_post_created import BcsDecodeError, decode_post_created_event
from app.chain.chain_types import CHECKPOINT_MARKER_EVENT, ChainEvent
from app.chain.checkpoint_event_extractor import (
    extract_post_created_events,
    format_address_padded,
    format_address_short,
    is_post_created_event,
    to_chain_events,
)
from app.chain.grpc_v2_client import GrpcV2Client, is_authenticated_events_disabled, is_subscription_unavailable
from app.network_config import NetworkProfile, ROOT_DIR

logger = structlog.get_logger()


def _resolve_event_stream_id_raw(profile: NetworkProfile) -> str:
    configured = (profile.grpc_sync.event_stream_id or "").strip()
    if configured:
        return configured
    package_id = os.getenv("MYSO_POC_PACKAGE_ID", "").strip()
    if package_id:
        return package_id
    objs = profile.objects or {}
    fallback = objs.get("poc_package") or objs.get("package") or profile.platform_package_address
    return str(fallback or "").strip()


def resolve_event_stream_id(profile: NetworkProfile) -> str:
    """Full padded package id for gRPC sync and event filtering."""
    return format_address_padded(_resolve_event_stream_id_raw(profile))


def resolve_checkpoint_stream_id(profile: NetworkProfile) -> str:
    """Short package id for grpc_sync_checkpoints DB keys."""
    return format_address_short(_resolve_event_stream_id_raw(profile))


def resolve_sync_mode(profile: NetworkProfile) -> str:
    if profile.grpc_sync.mock_mode:
        return "mock"
    mode = (profile.grpc_sync.sync_mode or "checkpoint_v2").strip().lower()
    if mode in ("mock", "checkpoint_v2", "authenticated_events"):
        return mode
    return "checkpoint_v2"


def resolve_sync_start_checkpoint(
    *,
    saved_checkpoint_sequence: int | None,
    configured_start: int | None,
) -> int:
    """Last checkpoint already processed (0 = none)."""
    if saved_checkpoint_sequence is not None:
        return int(saved_checkpoint_sequence)
    if configured_start is not None:
        return int(configured_start)
    return 0


def resolve_v2_next_sequence(
    start_checkpoint: int,
    configured_start: int | None,
) -> int:
    """First checkpoint sequence to fetch via GetCheckpoint."""
    if start_checkpoint > 0:
        return start_checkpoint + 1
    if configured_start is not None and configured_start > 0:
        return int(configured_start)
    return 1


def clamp_sequence_to_tip(next_seq: int, chain_tip: int) -> int:
    """Reset cursor when it is ahead of chain tip (e.g. after localnet restart)."""
    if next_seq > chain_tip:
        return 1 if chain_tip > 0 else 0
    return next_seq


class GrpcChainClient:
    def __init__(self, profile: NetworkProfile) -> None:
        self.profile = profile
        self.network = profile.network
        self.stream_id = resolve_event_stream_id(profile)
        self.sync_mode = resolve_sync_mode(profile)
        self._events_client: AuthenticatedEventsClient | None = None
        self._v2_client: GrpcV2Client | None = None
        self.last_highest_indexed_checkpoint: int = 0
        self.last_event_at_checkpoint: int | None = None
        self._auth_events_warning_logged = False

    def _events_client_instance(self) -> AuthenticatedEventsClient:
        if self._events_client is None:
            self._events_client = AuthenticatedEventsClient(self.profile)
        return self._events_client

    def _v2_client_instance(self) -> GrpcV2Client:
        if self._v2_client is None:
            self._v2_client = GrpcV2Client(self.profile)
        return self._v2_client

    async def stream_events(self, start_checkpoint: int = 0) -> AsyncIterator[ChainEvent]:
        if self.sync_mode == "mock":
            async for event in self._mock_fixture_stream(start_checkpoint):
                yield event
            return
        if self.sync_mode == "authenticated_events":
            async for event in self._authenticated_events_stream(start_checkpoint):
                yield event
            return
        async for event in self._checkpoint_v2_stream(start_checkpoint):
            yield event

    async def _mock_fixture_stream(self, start_checkpoint: int) -> AsyncIterator[ChainEvent]:
        fixture = self.profile.grpc_sync.fixture_path or "data/fixtures/post_created_events.json"
        path = Path(fixture)
        if not path.is_absolute():
            path = ROOT_DIR / path
        if not path.exists():
            logger.warning("Mock fixture missing", path=str(path))
            return
        events = json.loads(path.read_text(encoding="utf-8"))
        for raw in events:
            seq = int(raw.get("checkpoint_sequence", 0))
            if seq <= start_checkpoint:
                continue
            yield ChainEvent(
                event_type=str(raw.get("event_type", "PostCreatedEvent")),
                network=self.network,
                checkpoint_sequence=seq,
                transaction_digest=str(raw.get("transaction_digest", "")),
                payload=dict(raw.get("payload") or {}),
            )
            await asyncio.sleep(0.05)

    async def _yield_checkpoint_events(self, checkpoint, checkpoint_seq: int) -> AsyncIterator[ChainEvent]:
        if checkpoint is not None:
            extracted = extract_post_created_events(
                checkpoint,
                network=self.network,
                package_id=self.stream_id,
            )
            for event in to_chain_events(extracted, network=self.network):
                self.last_event_at_checkpoint = checkpoint_seq
                yield event
        yield ChainEvent(
            event_type=CHECKPOINT_MARKER_EVENT,
            network=self.network,
            checkpoint_sequence=checkpoint_seq,
            transaction_digest="",
            payload={},
        )

    async def _catch_up_checkpoints(
        self,
        client: GrpcV2Client,
        *,
        start_seq: int,
        end_seq: int,
    ) -> AsyncIterator[tuple[int, object]]:
        batch_size = max(1, self.profile.grpc_sync.checkpoint_catchup_batch_size)
        seq = start_seq
        while seq <= end_seq:
            batch_end = min(end_seq, seq + batch_size - 1)
            for current in range(seq, batch_end + 1):
                checkpoint = None
                try:
                    checkpoint = await client.fetch_checkpoint_async(current)
                except grpc.RpcError as exc:
                    if exc.code() == grpc.StatusCode.NOT_FOUND:
                        logger.debug(
                            "Checkpoint not found, advancing",
                            network=self.network,
                            checkpoint=current,
                        )
                    else:
                        raise
                yield current, checkpoint
            seq = batch_end + 1

    async def _poll_checkpoint_range(
        self,
        client: GrpcV2Client,
        *,
        start_seq: int,
    ) -> AsyncIterator[ChainEvent]:
        tip = await client.get_chain_tip_async()
        self.last_highest_indexed_checkpoint = tip
        if start_seq > tip:
            logger.warning(
                "Poll start ahead of chain tip — clamping",
                network=self.network,
                start_seq=start_seq,
                chain_tip=tip,
            )
            start_seq = 1 if tip > 0 else 0
        async for checkpoint_seq, checkpoint in self._catch_up_checkpoints(
            client,
            start_seq=start_seq,
            end_seq=tip,
        ):
            async for event in self._yield_checkpoint_events(checkpoint, checkpoint_seq):
                yield event

    async def _subscribe_checkpoints(
        self,
        client: GrpcV2Client,
        *,
        after_checkpoint: int,
    ) -> AsyncIterator[tuple[int, object]]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def producer() -> None:
            try:
                for cursor, checkpoint in client.iter_subscribe_checkpoints():
                    seq = int(checkpoint.sequence_number or cursor or 0)
                    if seq <= after_checkpoint:
                        continue
                    asyncio.run_coroutine_threadsafe(queue.put((seq, checkpoint)), loop).result()
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    async def _checkpoint_v2_stream(self, start_checkpoint: int) -> AsyncIterator[ChainEvent]:
        if not self.stream_id:
            logger.error(
                "Missing package stream_id — set MYSO_POC_PACKAGE_ID or grpc_sync.event_stream_id",
                network=self.network,
            )
            while True:
                await asyncio.sleep(self.profile.grpc_sync.poll_interval_seconds)

        client = self._v2_client_instance()
        next_seq = resolve_v2_next_sequence(
            start_checkpoint,
            self.profile.grpc_sync.start_checkpoint,
        )
        tip = await client.get_chain_tip_async()
        self.last_highest_indexed_checkpoint = max(self.last_highest_indexed_checkpoint, tip)
        clamped = clamp_sequence_to_tip(next_seq, tip)
        if clamped != next_seq:
            logger.warning(
                "Sync start ahead of chain tip — resetting for chain restart",
                network=self.network,
                configured_next_seq=next_seq,
                chain_tip=tip,
                saved_checkpoint=start_checkpoint,
                reset_to=clamped,
            )
            next_seq = clamped
        use_polling_only = False
        subscription_warned = False

        logger.info(
            "Starting v2 checkpoint sync",
            network=self.network,
            stream_id=self.stream_id,
            start_checkpoint=next_seq,
            grpc_target=client.target,
            tls=client.use_tls,
        )

        while True:
            try:
                if not use_polling_only:
                    try:
                        tip = await client.get_chain_tip_async()
                        self.last_highest_indexed_checkpoint = max(
                            self.last_highest_indexed_checkpoint,
                            tip,
                        )
                        clamped = clamp_sequence_to_tip(next_seq, tip)
                        if clamped != next_seq:
                            logger.warning(
                                "Sync cursor ahead of chain tip — resetting",
                                network=self.network,
                                cursor=next_seq,
                                chain_tip=tip,
                                reset_to=clamped,
                            )
                            next_seq = clamped

                        if next_seq <= tip:
                            async for checkpoint_seq, checkpoint in self._catch_up_checkpoints(
                                client,
                                start_seq=next_seq,
                                end_seq=tip,
                            ):
                                async for event in self._yield_checkpoint_events(checkpoint, checkpoint_seq):
                                    yield event
                                next_seq = checkpoint_seq + 1

                        tip = await client.get_chain_tip_async()
                        self.last_highest_indexed_checkpoint = max(
                            self.last_highest_indexed_checkpoint,
                            tip,
                        )
                        if next_seq > tip:
                            clamped = clamp_sequence_to_tip(next_seq, tip)
                            if clamped != next_seq:
                                logger.warning(
                                    "Skipping subscribe — cursor ahead of chain tip",
                                    network=self.network,
                                    cursor=next_seq,
                                    chain_tip=tip,
                                    reset_to=clamped,
                                )
                                next_seq = clamped
                            use_polling_only = True
                            continue

                        async for checkpoint_seq, checkpoint in self._subscribe_checkpoints(
                            client,
                            after_checkpoint=next_seq - 1,
                        ):
                            self.last_highest_indexed_checkpoint = max(
                                self.last_highest_indexed_checkpoint,
                                checkpoint_seq,
                            )
                            async for event in self._yield_checkpoint_events(checkpoint, checkpoint_seq):
                                yield event
                            next_seq = checkpoint_seq + 1
                        if not subscription_warned:
                            logger.warning(
                                "Checkpoint subscription ended — falling back to polling",
                                network=self.network,
                            )
                            subscription_warned = True
                        use_polling_only = True
                        continue
                    except Exception as exc:
                        if is_subscription_unavailable(exc):
                            if not subscription_warned:
                                logger.warning(
                                    "SubscribeCheckpoints unavailable — using GetCheckpoint polling",
                                    network=self.network,
                                    error=str(exc),
                                )
                                subscription_warned = True
                            use_polling_only = True
                        else:
                            raise

                async for event in self._poll_checkpoint_range(client, start_seq=next_seq):
                    yield event
                    if event.event_type == CHECKPOINT_MARKER_EVENT:
                        next_seq = event.checkpoint_sequence + 1
                await asyncio.sleep(self.profile.grpc_sync.poll_interval_seconds)
            except asyncio.CancelledError:
                raise
            except grpc.RpcError as exc:
                logger.warning(
                    "Checkpoint sync RPC error",
                    network=self.network,
                    error=str(exc),
                    code=str(exc.code()),
                )
                await asyncio.sleep(self.profile.grpc_sync.poll_interval_seconds)
            except Exception as exc:
                logger.warning(
                    "Checkpoint sync error",
                    network=self.network,
                    error=str(exc),
                )
                await asyncio.sleep(self.profile.grpc_sync.poll_interval_seconds)

    async def _authenticated_events_stream(self, start_checkpoint: int) -> AsyncIterator[ChainEvent]:
        if not self.stream_id:
            logger.error(
                "Missing event stream_id — set MYSO_POC_PACKAGE_ID or grpc_sync.event_stream_id",
                network=self.network,
            )
            while True:
                await asyncio.sleep(self.profile.grpc_sync.poll_interval_seconds)

        client = self._events_client_instance()
        api_start = start_checkpoint + 1 if start_checkpoint > 0 else int(
            self.profile.grpc_sync.start_checkpoint or 0
        )
        logger.info(
            "Starting authenticated events stream",
            network=self.network,
            stream_id=self.stream_id,
            start_checkpoint=api_start,
            grpc_target=client.target,
            tls=client.use_tls,
        )

        while True:
            try:
                batch = await client.iter_authenticated_events_async(
                    stream_id=self.stream_id,
                    start_checkpoint=api_start,
                    page_size=self.profile.grpc_sync.batch_size,
                    max_pagination_iterations=self.profile.grpc_sync.max_pagination_iterations,
                )
                self.last_highest_indexed_checkpoint = client.last_highest_indexed_checkpoint

                if not batch:
                    await asyncio.sleep(self.profile.grpc_sync.poll_interval_seconds)
                    continue

                for auth_event in batch:
                    if not is_post_created_event(auth_event.module, auth_event.event_type):
                        continue
                    try:
                        decoded = decode_post_created_event(auth_event.contents)
                    except BcsDecodeError as exc:
                        logger.warning(
                            "PostCreatedEvent BCS decode failed",
                            network=self.network,
                            error=str(exc),
                            event_type=auth_event.event_type,
                        )
                        continue

                    payload = decoded.to_payload()
                    self.last_event_at_checkpoint = auth_event.checkpoint
                    yield ChainEvent(
                        event_type="PostCreatedEvent",
                        network=self.network,
                        checkpoint_sequence=auth_event.checkpoint,
                        transaction_digest=(
                            f"cp{auth_event.checkpoint}_tx{auth_event.transaction_idx}_ev{auth_event.event_idx}"
                        ),
                        payload=payload,
                    )

                last_checkpoint = batch[-1].checkpoint
                api_start = last_checkpoint + 1
            except asyncio.CancelledError:
                raise
            except grpc.RpcError as exc:
                if is_authenticated_events_disabled(exc) and not self._auth_events_warning_logged:
                    logger.warning(
                        "Authenticated events indexing disabled — set grpc_sync.sync_mode: checkpoint_v2",
                        network=self.network,
                    )
                    self._auth_events_warning_logged = True
                await asyncio.sleep(self.profile.grpc_sync.poll_interval_seconds)
            except Exception as exc:
                logger.warning(
                    "Authenticated events poll failed",
                    network=self.network,
                    stream_id=self.stream_id,
                    error=str(exc),
                )
                await asyncio.sleep(self.profile.grpc_sync.poll_interval_seconds)
