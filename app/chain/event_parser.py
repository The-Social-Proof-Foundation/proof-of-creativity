"""Parse PostCreatedEvent and related chain events."""

from __future__ import annotations

import json
from typing import Any


def parse_post_metadata_commitments(metadata_json: str | None) -> dict[str, str] | None:
    """Extract PoC commitment fields embedded in post metadata_json by the client."""
    if not metadata_json or not str(metadata_json).strip():
        return None
    try:
        data = json.loads(metadata_json)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None

    content = data.get("content_commitment") or data.get("contentCommitment")
    fingerprint = data.get("observed_fingerprint_commitment") or data.get("observedFingerprintCommitment")
    if not content or not fingerprint:
        return None

    def _strip_hex(value: str) -> str:
        text = str(value).strip()
        return text[2:] if text.lower().startswith("0x") else text

    return {
        "content_commitment": _strip_hex(content),
        "observed_fingerprint_commitment": _strip_hex(fingerprint),
        "asset_id": str(data.get("asset_id") or data.get("assetId") or ""),
    }


def parse_post_created(payload: dict[str, Any]) -> dict[str, Any] | None:
    post_id = payload.get("post_id") or payload.get("post") or payload.get("id")
    if not post_id:
        return None
    enable_poc = payload.get("enable_poc", True)
    if isinstance(enable_poc, str):
        enable_poc = enable_poc.lower() in ("true", "1", "yes")

    owner = payload.get("owner") or payload.get("creator") or payload.get("creator_address")
    creator = payload.get("creator") or owner

    media_urls = payload.get("media_urls")
    if media_urls is None:
        media_urls = []
    elif isinstance(media_urls, str):
        media_urls = [media_urls]

    media_types = payload.get("media_types") or payload.get("media_type") or []
    if isinstance(media_types, int):
        media_types = [media_types]
    media_asset_ids = payload.get("media_asset_ids") or []
    if isinstance(media_asset_ids, str):
        media_asset_ids = [media_asset_ids]
    return {
        "post_id": str(post_id),
        "owner": str(owner) if owner else None,
        "creator": str(creator) if creator else None,
        "enable_poc": bool(enable_poc),
        "media_urls": [str(u) for u in media_urls if u],
        "media_types": [int(t) for t in media_types] if media_types else [],
        "media_asset_ids": [str(a) for a in media_asset_ids if a],
        "composition_status": payload.get("composition_status"),
        "monetization_status": payload.get("monetization_status"),
        "spt_id": payload.get("spt_id"),
        "metadata_json": payload.get("metadata_json"),
    }


def infer_primary_media_type(media_urls: list[str], media_types: list[int]) -> int:
    """Pick the primary media type for post media (prefer video when HLS present)."""
    for idx, url in enumerate(media_urls):
        if any(ext in url.lower() for ext in (".m3u8", ".mp4", ".mov", ".webm", ".mkv")):
            return infer_media_type(url, idx, media_types)
    if media_types:
        return int(media_types[0])
    if media_urls:
        return infer_media_type(media_urls[0], 0, media_types)
    return 1


def infer_media_type(url: str, index: int, media_types: list[int]) -> int:
    if index < len(media_types):
        return int(media_types[index])
    lower = url.lower()
    if any(lower.endswith(ext) for ext in (".mp4", ".mov", ".webm", ".mkv", ".m3u8")):
        return 2
    if ".m3u8" in lower:
        return 2
    if any(lower.endswith(ext) for ext in (".mp3", ".wav", ".aac", ".flac", ".m4a", ".ogg")):
        return 3
    return 1
