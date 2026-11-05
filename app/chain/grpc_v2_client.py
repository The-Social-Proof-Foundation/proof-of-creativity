"""v2 gRPC client for checkpoint-based chain sync."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import Any

import grpc
import structlog
from google.protobuf import field_mask_pb2

from app.chain.grpc_transport import parse_grpc_target
from app.chain.proto.myso.rpc.v2 import (
    checkpoint_pb2,
    ledger_service_pb2,
    ledger_service_pb2_grpc,
    subscription_service_pb2,
    subscription_service_pb2_grpc,
)
from app.network_config import NetworkProfile

logger = structlog.get_logger()

CHECKPOINT_EVENTS_READ_MASK = field_mask_pb2.FieldMask(
    paths=[
        "sequence_number",
        "transactions.digest",
        "transactions.checkpoint",
        "transactions.events.events.package_id",
        "transactions.events.events.module",
        "transactions.events.events.event_type",
        "transactions.events.events.contents.value",
    ]
)


class GrpcV2Client:
    def __init__(self, profile: NetworkProfile) -> None:
        self.profile = profile
        self.target, self.use_tls = parse_grpc_target(profile.grpc_url, profile.grpc_tls)
        self._channel: grpc.Channel | None = None
        self._ledger: ledger_service_pb2_grpc.LedgerServiceStub | None = None
        self._subscription: subscription_service_pb2_grpc.SubscriptionServiceStub | None = None
        self.last_chain_tip: int = 0

    def _ensure_channel(self) -> grpc.Channel:
        if self._channel is not None:
            return self._channel
        if self.use_tls:
            credentials = grpc.ssl_channel_credentials()
            self._channel = grpc.secure_channel(self.target, credentials)
        else:
            self._channel = grpc.insecure_channel(self.target)
        self._ledger = ledger_service_pb2_grpc.LedgerServiceStub(self._channel)
        self._subscription = subscription_service_pb2_grpc.SubscriptionServiceStub(self._channel)
        return self._channel

    def close(self) -> None:
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._ledger = None
            self._subscription = None

    def get_service_info(self) -> ledger_service_pb2.GetServiceInfoResponse:
        self._ensure_channel()
        assert self._ledger is not None
        response = self._ledger.GetServiceInfo(
            ledger_service_pb2.GetServiceInfoRequest(),
            timeout=30.0,
        )
        tip = int(response.checkpoint_height or 0)
        self.last_chain_tip = tip
        return response

    def get_chain_tip(self) -> int:
        return int(self.get_service_info().checkpoint_height or 0)

    def fetch_checkpoint(
        self,
        sequence_number: int,
        *,
        read_mask: field_mask_pb2.FieldMask | None = None,
    ) -> checkpoint_pb2.Checkpoint | None:
        self._ensure_channel()
        assert self._ledger is not None
        request = ledger_service_pb2.GetCheckpointRequest(
            sequence_number=sequence_number,
            read_mask=read_mask or CHECKPOINT_EVENTS_READ_MASK,
        )
        response = self._ledger.GetCheckpoint(request, timeout=60.0)
        if not response.HasField("checkpoint"):
            return None
        return response.checkpoint

    def iter_subscribe_checkpoints(
        self,
        *,
        read_mask: field_mask_pb2.FieldMask | None = None,
    ) -> Iterator[tuple[int, checkpoint_pb2.Checkpoint]]:
        self._ensure_channel()
        assert self._subscription is not None
        request = subscription_service_pb2.SubscribeCheckpointsRequest(
            read_mask=read_mask or CHECKPOINT_EVENTS_READ_MASK,
        )
        stream = self._subscription.SubscribeCheckpoints(request, timeout=None)
        for message in stream:
            if not message.HasField("checkpoint"):
                continue
            cursor = int(message.cursor or message.checkpoint.sequence_number or 0)
            tip = int(message.checkpoint.sequence_number or cursor)
            self.last_chain_tip = max(self.last_chain_tip, tip)
            yield cursor, message.checkpoint

    async def get_chain_tip_async(self) -> int:
        return await asyncio.to_thread(self.get_chain_tip)

    async def fetch_checkpoint_async(
        self,
        sequence_number: int,
        *,
        read_mask: field_mask_pb2.FieldMask | None = None,
    ) -> checkpoint_pb2.Checkpoint | None:
        return await asyncio.to_thread(
            self.fetch_checkpoint,
            sequence_number,
            read_mask=read_mask,
        )

    async def iter_subscribe_checkpoints_async(
        self,
        *,
        read_mask: field_mask_pb2.FieldMask | None = None,
    ) -> AsyncIterator[tuple[int, checkpoint_pb2.Checkpoint]]:
        queue: asyncio.Queue[tuple[int, checkpoint_pb2.Checkpoint] | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _producer() -> None:
            try:
                for item in self.iter_subscribe_checkpoints(read_mask=read_mask):
                    loop.call_soon_threadsafe(queue.put_nowait, item)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)  # type: ignore[arg-type]
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        asyncio.create_task(asyncio.to_thread(_producer))
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item


def is_subscription_unavailable(exc: BaseException) -> bool:
    if not isinstance(exc, grpc.RpcError):
        return False
    details = str(getattr(exc, "details", lambda: "")() or "").lower()
    return "subscription service not enabled" in details or exc.code() == grpc.StatusCode.UNIMPLEMENTED


def is_authenticated_events_disabled(exc: BaseException) -> bool:
    if not isinstance(exc, grpc.RpcError):
        return False
    details = str(getattr(exc, "details", lambda: "")() or "").lower()
    return "authenticated events indexing is disabled" in details
