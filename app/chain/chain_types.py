"""Shared chain event types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChainEvent:
    event_type: str
    network: str
    checkpoint_sequence: int
    transaction_digest: str
    payload: dict


CHECKPOINT_MARKER_EVENT = "__checkpoint__"
