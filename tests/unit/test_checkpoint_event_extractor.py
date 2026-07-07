"""Unit tests for checkpoint event extraction."""

from __future__ import annotations

from app.chain.checkpoint_event_extractor import (
    addresses_match,
    extract_post_created_events,
    format_address_padded,
    format_address_short,
    normalize_address,
    to_chain_events,
)
from app.chain.proto.myso.rpc.v2 import bcs_pb2, checkpoint_pb2, event_pb2, executed_transaction_pb2
from tests.unit.test_bcs_post_created import encode_post_created_with_organization


def _build_checkpoint_with_post_created(*, package_id: str, contents: bytes) -> checkpoint_pb2.Checkpoint:
    tx_events = event_pb2.TransactionEvents()
    tx_events.events.append(
        event_pb2.Event(
            package_id=package_id,
            module="post",
            event_type=f"{package_id}::post::PostCreatedEvent",
            contents=bcs_pb2.Bcs(value=contents),
        )
    )
    tx = executed_transaction_pb2.ExecutedTransaction(
        digest="abc123",
        checkpoint=42,
        events=tx_events,
    )
    cp = checkpoint_pb2.Checkpoint(sequence_number=42)
    cp.transactions.append(tx)
    return cp


def test_normalize_address_padding():
    assert normalize_address("0x50c1") == normalize_address(
        "0x00000000000000000000000000000000000000000000000000000000000050c1"
    )
    assert addresses_match("0x50c1", "0x00000000000000000000000000000000000000000000000000000000000050c1")


def test_format_address_short_and_padded():
    short = format_address_short("0x00000000000000000000000000000000000000000000000000000000000050c1")
    padded = format_address_padded("0x50c1")
    assert short == "0x50c1"
    assert len(padded) == 66
    assert padded.endswith("50c1")


def test_extract_post_created_from_checkpoint():
    package_id = "0x00000000000000000000000000000000000000000000000000000000000050c1"
    contents = encode_post_created_with_organization(
        media_urls=["https://example.com/a.jpg"],
        enable_poc=True,
    )
    checkpoint = _build_checkpoint_with_post_created(package_id=package_id, contents=contents)
    extracted = extract_post_created_events(
        checkpoint,
        network="localnet",
        package_id=package_id,
    )
    assert len(extracted) == 1
    assert extracted[0].transaction_digest == "abc123"
    assert extracted[0].payload["enable_poc"] is True
    assert extracted[0].payload["media_urls"] == ["https://example.com/a.jpg"]

    chain_events = to_chain_events(extracted, network="localnet")
    assert chain_events[0].event_type == "PostCreatedEvent"
    assert chain_events[0].checkpoint_sequence == 42


def test_extract_skips_wrong_package():
    package_id = "0x00000000000000000000000000000000000000000000000000000000000050c1"
    other = "0x00000000000000000000000000000000000000000000000000000000000000aa"
    contents = encode_post_created_with_organization(media_urls=["https://x/y.jpg"])
    checkpoint = _build_checkpoint_with_post_created(package_id=other, contents=contents)
    assert extract_post_created_events(checkpoint, network="localnet", package_id=package_id) == []
