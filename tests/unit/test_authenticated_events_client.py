"""Unit tests for authenticated events gRPC client."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.chain.authenticated_events_client import AuthenticatedEventsClient
from app.chain.grpc_transport import parse_grpc_target
from app.chain.proto.myso.rpc.alpha import event_service_pb2
from app.chain.proto.myso.rpc.v2 import bcs_pb2, event_pb2
from app.network_config import load_network_profile


def test_parse_grpc_target_https():
    target, tls = parse_grpc_target("https://fullnode.testnet.mysocial.io:443", True)
    assert target == "fullnode.testnet.mysocial.io:443"
    assert tls is True


def test_list_authenticated_events_pagination():
    profile = load_network_profile("localnet")
    client = AuthenticatedEventsClient(profile)

    page1_event = event_service_pb2.AuthenticatedEvent(
        checkpoint=10,
        transaction_idx=1,
        event_idx=0,
        stream_id="0x50c1",
        event=event_pb2.Event(
            package_id="0x50c1",
            module="post",
            event_type="0x50c1::post::PostCreatedEvent",
            sender="0x2",
            contents=bcs_pb2.Bcs(value=b"\x01"),
        ),
    )
    page1 = event_service_pb2.ListAuthenticatedEventsResponse(
        events=[page1_event],
        highest_indexed_checkpoint=42,
        next_page_token=b"token",
    )
    page2 = event_service_pb2.ListAuthenticatedEventsResponse(
        events=[],
        highest_indexed_checkpoint=42,
    )

    stub = MagicMock()
    stub.ListAuthenticatedEvents.side_effect = [page1, page2]
    client._stub = stub

    events = list(
        client.iter_authenticated_events(
            stream_id="0x50c1",
            start_checkpoint=0,
            page_size=1000,
            max_pagination_iterations=5,
        )
    )
    assert len(events) == 1
    assert events[0].checkpoint == 10
    assert events[0].module == "post"
    assert client.last_highest_indexed_checkpoint == 42
    assert stub.ListAuthenticatedEvents.call_count == 2


def test_peek_highest_indexed_checkpoint():
    profile = load_network_profile("localnet")
    client = AuthenticatedEventsClient(profile)
    stub = MagicMock()
    stub.ListAuthenticatedEvents.return_value = event_service_pb2.ListAuthenticatedEventsResponse(
        events=[],
        highest_indexed_checkpoint=999,
    )
    client._stub = stub
    assert client.peek_highest_indexed_checkpoint("0x50c1") == 999
