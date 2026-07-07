"""Parse PostCreatedEvent and related chain events."""

from __future__ import annotations

from typing import Any


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
    return {
        "post_id": str(post_id),
        "owner": str(owner) if owner else None,
        "creator": str(creator) if creator else None,
        "enable_poc": bool(enable_poc),
        "media_urls": [str(u) for u in media_urls if u],
        "media_types": [int(t) for t in media_types] if media_types else [],
        "spt_id": payload.get("spt_id"),
    }


def infer_media_type(url: str, index: int, media_types: list[int]) -> int:
    if index < len(media_types):
        return int(media_types[index])
    lower = url.lower()
    if any(lower.endswith(ext) for ext in (".mp4", ".mov", ".webm", ".mkv")):
        return 2
    if any(lower.endswith(ext) for ext in (".mp3", ".wav", ".aac", ".flac", ".m4a", ".ogg")):
        return 3
    return 1
