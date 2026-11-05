"""BCS decoder for media_asset::MediaResolutionRequestedEvent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.chain.bcs_post_created import BcsDecodeError, BcsReader


@dataclass
class MediaResolutionRequestedDecoded:
    request_id: str
    content_commitment: bytes
    observed_fingerprint_commitment: bytes
    media_type: int
    submitter: str
    timestamp: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "content_commitment": self.content_commitment.hex(),
            "observed_fingerprint_commitment": self.observed_fingerprint_commitment.hex(),
            "media_type": self.media_type,
            "submitter": self.submitter,
            "timestamp": self.timestamp,
        }


def decode_media_resolution_requested(contents: bytes) -> MediaResolutionRequestedDecoded:
    if not contents:
        raise BcsDecodeError("empty contents")
    reader = BcsReader(contents)
    request_id = reader.read_address()
    content_commitment = reader.read_vec_u8()
    observed_fingerprint_commitment = reader.read_vec_u8()
    media_type = reader.read_u8()
    submitter = reader.read_address()
    timestamp = reader.read_u64()
    if reader.remaining != 0:
        raise BcsDecodeError(f"trailing bytes: {reader.remaining}")
    return MediaResolutionRequestedDecoded(
        request_id=request_id,
        content_commitment=content_commitment,
        observed_fingerprint_commitment=observed_fingerprint_commitment,
        media_type=media_type,
        submitter=submitter,
        timestamp=timestamp,
    )
