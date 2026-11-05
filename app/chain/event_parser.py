"""Parse PostCreatedEvent and related chain events."""

from __future__ import annotations

import json
from typing import Any


def _strip_hex(value: str) -> str:
    text = str(value).strip()
    return text[2:] if text.lower().startswith("0x") else text


def _metadata_dict(metadata_json: str | None) -> dict[str, Any] | None:
    if not metadata_json or not str(metadata_json).strip():
        return None
    try:
        data = json.loads(metadata_json)
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def parse_post_metadata_commitments(metadata_json: str | None) -> dict[str, str] | None:
    """Extract PoC commitment fields embedded in post metadata_json by the client."""
    data = _metadata_dict(metadata_json)
    if not data:
        return None

    content = data.get("content_commitment") or data.get("contentCommitment")
    fingerprint = data.get("observed_fingerprint_commitment") or data.get("observedFingerprintCommitment")
    if not content or not fingerprint:
        return None

    return {
        "content_commitment": _strip_hex(content),
        "observed_fingerprint_commitment": _strip_hex(fingerprint),
        "asset_id": str(data.get("asset_id") or data.get("assetId") or ""),
    }


def parse_post_media_clips(
    media_urls: list[str],
    media_types: list[int],
    metadata_json: str | None,
) -> list[dict[str, Any]]:
    """Build per-clip resolve inputs from metadata (array or top-level pair = clip 0 / primary)."""
    data = _metadata_dict(metadata_json) or {}
    raw_clips = data.get("media_commitments") or data.get("mediaCommitments")
    clips: list[dict[str, Any]] = []
    if isinstance(raw_clips, list) and raw_clips:
        for index, item in enumerate(raw_clips):
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or item.get("media_url") or item.get("mediaUrl") or "")
            if not url and index < len(media_urls):
                url = media_urls[index]
            content = item.get("content_commitment") or item.get("contentCommitment")
            fingerprint = (
                item.get("observed_fingerprint_commitment") or item.get("observedFingerprintCommitment")
            )
            asset_id = str(item.get("media_asset_id") or item.get("mediaAssetId") or "")
            if not url and not content and not asset_id:
                continue
            media_type = item.get("media_type") or item.get("mediaType")
            if media_type is None:
                media_type = infer_media_type(url, index, media_types) if url else 1
            clips.append(
                {
                    "index": index,
                    "url": url,
                    "media_type": int(media_type),
                    "content_commitment": _strip_hex(content) if content else "",
                    "observed_fingerprint_commitment": _strip_hex(fingerprint) if fingerprint else "",
                    "media_asset_id": asset_id,
                    "storage_asset_id": str(item.get("asset_id") or item.get("assetId") or ""),
                }
            )
        return clips

    commitments = parse_post_metadata_commitments(metadata_json)
    if not commitments or not media_urls:
        return []
    index = 0
    primary = infer_primary_media_type(media_urls, media_types)
    for idx, url in enumerate(media_urls):
        if infer_media_type(url, idx, media_types) == primary:
            index = idx
            break
    url = media_urls[index]
    return [
        {
            "index": index,
            "url": url,
            "media_type": infer_media_type(url, index, media_types),
            "content_commitment": commitments["content_commitment"],
            "observed_fingerprint_commitment": commitments["observed_fingerprint_commitment"],
            "media_asset_id": "",
            "storage_asset_id": commitments.get("asset_id") or "",
        }
    ]


def plan_post_media_jobs(
    *,
    media_urls: list[str],
    media_types: list[int],
    media_asset_ids: list[str],
    metadata_json: str | None,
    enable_poc: bool = True,
) -> dict[str, Any]:
    """Decide resolve_post_media vs compose vs legacy analyze_post. Existing ids do not skip new clips."""
    clips = parse_post_media_clips(media_urls, media_types, metadata_json)
    existing = [str(a) for a in media_asset_ids if a]
    for clip in clips:
        asset_id = str(clip.get("media_asset_id") or "")
        if asset_id and asset_id not in existing:
            existing.append(asset_id)
    unresolved = [
        clip
        for clip in clips
        if clip.get("content_commitment")
        and clip.get("observed_fingerprint_commitment")
        and not clip.get("media_asset_id")
    ]
    if unresolved:
        return {
            "action": "resolve_post_media",
            "clips": unresolved,
            "existing_media_asset_ids": existing,
        }
    if existing:
        return {"action": "analyze_composition", "media_asset_ids": existing}
    if enable_poc and media_urls:
        return {"action": "analyze_post"}
    return {"action": "skip"}


def parse_post_audio_metadata(metadata_json: str | None) -> dict:
    """Extract the canonical edit recipe and per-asset roles; ignores unrelated metadata."""
    if not metadata_json or not str(metadata_json).strip():
        return {}
    try:
        data = json.loads(str(metadata_json))
    except (TypeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    recipe = data.get("audioEditRecipe")
    roles = data.get("mediaAssetRoles")
    return {
        "audio_edit_recipe": recipe if isinstance(recipe, dict) else None,
        "media_asset_roles": roles if isinstance(roles, dict) else {},
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
    platform_id = payload.get("platform_id") or payload.get("platformId")
    return {
        "post_id": str(post_id),
        "owner": str(owner) if owner else None,
        "creator": str(creator) if creator else None,
        "platform_id": str(platform_id) if platform_id else None,
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
