"""Extract PostCreatedEvent chain events from v2 checkpoint protos."""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from app.chain.bcs_post_created import BcsDecodeError, decode_post_created_event
from app.chain.bcs_media_resolution import decode_media_resolution_requested
from app.chain.chain_types import CHECKPOINT_MARKER_EVENT, ChainEvent
from app.chain.proto.myso.rpc.v2 import checkpoint_pb2

logger = structlog.get_logger()

POST_CREATED_SUFFIX = "::PostCreatedEvent"
MEDIA_RESOLUTION_REQUESTED_SUFFIX = "::MediaResolutionRequestedEvent"


def normalize_address(value: str) -> str:
    raw = (value or "").strip().lower()
    if not raw:
        return ""
    if raw.startswith("0x"):
        raw = raw[2:]
    return raw.zfill(64)


def format_address_padded(value: str) -> str:
    """Full 32-byte address for gRPC event filtering (66 chars with 0x)."""
    hex_part = normalize_address(value)
    if not hex_part:
        return ""
    return f"0x{hex_part}"


def format_address_short(value: str) -> str:
    """Canonical short address for DB keys (e.g. 0x50c1)."""
    hex_part = normalize_address(value)
    if not hex_part:
        return ""
    short = hex_part.lstrip("0") or "0"
    return f"0x{short}"


def addresses_match(left: str, right: str) -> bool:
    return normalize_address(left) == normalize_address(right)


def is_post_created_event(module: str, event_type: str) -> bool:
    return module == "post" and event_type.endswith(POST_CREATED_SUFFIX)


def _is_post_created_event(module: str, event_type: str) -> bool:
    return is_post_created_event(module, event_type)


@dataclass
class ExtractedCheckpointEvent:
    checkpoint_sequence: int
    transaction_digest: str
    transaction_idx: int
    event_idx: int
    payload: dict


def is_media_resolution_requested_event(module: str, event_type: str) -> bool:
    return module == "media_asset" and event_type.endswith(MEDIA_RESOLUTION_REQUESTED_SUFFIX)


def extract_chain_events(
    checkpoint: checkpoint_pb2.Checkpoint,
    *,
    network: str,
    package_id: str,
) -> list[ExtractedCheckpointEvent]:
    """Walk checkpoint transactions and return decoded PoC/media_asset events."""
    results: list[ExtractedCheckpointEvent] = []
    checkpoint_seq = int(checkpoint.sequence_number or 0)
    normalized_package = normalize_address(package_id)

    for tx_idx, transaction in enumerate(checkpoint.transactions):
        digest = str(transaction.digest or "")
        if not transaction.HasField("events"):
            continue
        for event_idx, event in enumerate(transaction.events.events):
            module = str(event.module or "")
            event_type = str(event.event_type or "")
            event_package = str(event.package_id or "")
            if normalized_package and not addresses_match(event_package, normalized_package):
                continue
            contents = b""
            if event.HasField("contents") and event.contents.value:
                contents = bytes(event.contents.value)

            if _is_post_created_event(module, event_type):
                try:
                    decoded = decode_post_created_event(contents)
                except BcsDecodeError as exc:
                    logger.warning(
                        "PostCreatedEvent BCS decode failed",
                        network=network,
                        checkpoint=checkpoint_seq,
                        digest=digest,
                        error=str(exc),
                    )
                    continue
                results.append(
                    ExtractedCheckpointEvent(
                        checkpoint_sequence=checkpoint_seq,
                        transaction_digest=digest,
                        transaction_idx=tx_idx,
                        event_idx=event_idx,
                        payload={"event_type": "PostCreatedEvent", **decoded.to_payload()},
                    )
                )
                continue

            if is_media_resolution_requested_event(module, event_type):
                try:
                    decoded = decode_media_resolution_requested(contents)
                except BcsDecodeError as exc:
                    logger.warning(
                        "MediaResolutionRequestedEvent BCS decode failed",
                        network=network,
                        checkpoint=checkpoint_seq,
                        digest=digest,
                        error=str(exc),
                    )
                    continue
                results.append(
                    ExtractedCheckpointEvent(
                        checkpoint_sequence=checkpoint_seq,
                        transaction_digest=digest,
                        transaction_idx=tx_idx,
                        event_idx=event_idx,
                        payload={"event_type": "MediaResolutionRequestedEvent", **decoded.to_payload()},
                    )
                )
    return results


def extract_post_created_events(
    checkpoint: checkpoint_pb2.Checkpoint,
    *,
    network: str,
    package_id: str,
) -> list[ExtractedCheckpointEvent]:
    """Backward-compatible wrapper returning PostCreatedEvent payloads only."""
    return [
        item
        for item in extract_chain_events(checkpoint, network=network, package_id=package_id)
        if item.payload.get("event_type") == "PostCreatedEvent"
    ]


def to_chain_events(
    extracted: list[ExtractedCheckpointEvent],
    *,
    network: str,
) -> list[ChainEvent]:
    out: list[ChainEvent] = []
    for item in extracted:
        event_type = str(item.payload.get("event_type") or "PostCreatedEvent")
        payload = {k: v for k, v in item.payload.items() if k != "event_type"}
        out.append(
            ChainEvent(
                event_type=event_type,
                network=network,
                checkpoint_sequence=item.checkpoint_sequence,
                transaction_digest=item.transaction_digest,
                payload=payload,
                event_idx=item.event_idx,
            )
        )
    return out
