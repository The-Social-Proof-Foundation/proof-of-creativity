"""gRPC client for EventService.ListAuthenticatedEvents."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Iterator

import grpc
import structlog

from app.chain.grpc_transport import parse_grpc_target
from app.chain.proto.myso.rpc.alpha import event_service_pb2, event_service_pb2_grpc
from app.network_config import NetworkProfile

logger = structlog.get_logger()

MAX_PAGE_SIZE = 1000


@dataclass
class AuthenticatedChainEvent:
    checkpoint: int
    transaction_idx: int
    event_idx: int
    stream_id: str
    package_id: str
    module: str
    event_type: str
    sender: str
    contents: bytes


@dataclass
class AuthenticatedEventsPage:
    events: list[AuthenticatedChainEvent]
    highest_indexed_checkpoint: int
    next_page_token: bytes | None


class AuthenticatedEventsClient:
    def __init__(self, profile: NetworkProfile) -> None:
        self.profile = profile
        self.target, self.use_tls = parse_grpc_target(profile.grpc_url, profile.grpc_tls)
        self._channel: grpc.Channel | None = None
        self._stub: event_service_pb2_grpc.EventServiceStub | None = None
        self.last_highest_indexed_checkpoint: int = 0

    def _ensure_stub(self) -> event_service_pb2_grpc.EventServiceStub:
        if self._stub is not None:
            return self._stub
        if self.use_tls:
            credentials = grpc.ssl_channel_credentials()
            self._channel = grpc.secure_channel(self.target, credentials)
        else:
            self._channel = grpc.insecure_channel(self.target)
        self._stub = event_service_pb2_grpc.EventServiceStub(self._channel)
        return self._stub

    def close(self) -> None:
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None

    def list_authenticated_events(
        self,
        *,
        stream_id: str,
        start_checkpoint: int,
        page_size: int,
        page_token: bytes | None = None,
    ) -> AuthenticatedEventsPage:
        stub = self._ensure_stub()
        request = event_service_pb2.ListAuthenticatedEventsRequest(
            stream_id=stream_id,
            start_checkpoint=start_checkpoint,
            page_size=min(max(1, page_size), MAX_PAGE_SIZE),
        )
        if page_token:
            request.page_token = page_token

        response = stub.ListAuthenticatedEvents(request, timeout=30.0)
        highest = int(response.highest_indexed_checkpoint or 0)
        self.last_highest_indexed_checkpoint = highest

        events: list[AuthenticatedChainEvent] = []
        for auth_event in response.events:
            ev = auth_event.event
            if not ev:
                continue
            contents = b""
            if ev.contents and ev.contents.value:
                contents = bytes(ev.contents.value)
            events.append(
                AuthenticatedChainEvent(
                    checkpoint=int(auth_event.checkpoint or 0),
                    transaction_idx=int(auth_event.transaction_idx or 0),
                    event_idx=int(auth_event.event_idx or 0),
                    stream_id=str(auth_event.stream_id or stream_id),
                    package_id=str(ev.package_id or ""),
                    module=str(ev.module or ""),
                    event_type=str(ev.event_type or ""),
                    sender=str(ev.sender or ""),
                    contents=contents,
                )
            )

        next_token = bytes(response.next_page_token) if response.next_page_token else None
        return AuthenticatedEventsPage(
            events=events,
            highest_indexed_checkpoint=highest,
            next_page_token=next_token if next_token else None,
        )

    def iter_authenticated_events(
        self,
        *,
        stream_id: str,
        start_checkpoint: int,
        page_size: int,
        max_pagination_iterations: int,
    ) -> Iterator[AuthenticatedChainEvent]:
        page_token: bytes | None = None
        api_start = start_checkpoint
        iterations = 0

        while iterations < max_pagination_iterations:
            iterations += 1
            page = self.list_authenticated_events(
                stream_id=stream_id,
                start_checkpoint=api_start,
                page_size=page_size,
                page_token=page_token,
            )
            if not page.events:
                break
            for event in page.events:
                yield event
            if not page.next_page_token:
                break
            page_token = page.next_page_token

        if iterations >= max_pagination_iterations:
            logger.warning(
                "Authenticated events pagination limit reached",
                network=self.profile.network,
                stream_id=stream_id,
                iterations=iterations,
            )

    def peek_highest_indexed_checkpoint(self, stream_id: str) -> int:
        page = self.list_authenticated_events(
            stream_id=stream_id,
            start_checkpoint=0,
            page_size=1,
        )
        return page.highest_indexed_checkpoint

    async def list_authenticated_events_async(
        self,
        *,
        stream_id: str,
        start_checkpoint: int,
        page_size: int,
        page_token: bytes | None = None,
    ) -> AuthenticatedEventsPage:
        return await asyncio.to_thread(
            self.list_authenticated_events,
            stream_id=stream_id,
            start_checkpoint=start_checkpoint,
            page_size=page_size,
            page_token=page_token,
        )

    async def iter_authenticated_events_async(
        self,
        *,
        stream_id: str,
        start_checkpoint: int,
        page_size: int,
        max_pagination_iterations: int,
    ) -> list[AuthenticatedChainEvent]:
        return await asyncio.to_thread(
            lambda: list(
                self.iter_authenticated_events(
                    stream_id=stream_id,
                    start_checkpoint=start_checkpoint,
                    page_size=page_size,
                    max_pagination_iterations=max_pagination_iterations,
                )
            ),
        )
