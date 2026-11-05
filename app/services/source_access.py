"""Platform-neutral media source access (public HTTP default + pluggable private leases)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from app.services.dripdrop_video_client import extract_asset_id_from_hls, get_dripdrop_video_client

ADAPTER_PUBLIC_HTTP = "public_http"
ADAPTER_DRIPDROP_HMAC = "dripdrop_hmac"

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "platforms.yaml"


def _normalize_platform_id(value: str | None) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("0x"):
        body = text[2:].lstrip("0") or "0"
        return "0x" + body
    return text


@lru_cache(maxsize=1)
def _adapter_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    if _CONFIG_PATH.exists():
        data = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) or {}
        for entry in data.get("adapters") or []:
            if not isinstance(entry, dict):
                continue
            platform_id = _normalize_platform_id(str(entry.get("platform_id") or ""))
            adapter = str(entry.get("adapter") or "").strip()
            if platform_id and adapter:
                mapping[platform_id] = adapter
    raw = os.getenv("POC_PLATFORM_ADAPTERS", "").strip()
    for part in raw.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        platform_id, adapter = part.split(":", 1)
        platform_id = _normalize_platform_id(platform_id)
        adapter = adapter.strip()
        if platform_id and adapter:
            mapping[platform_id] = adapter
    dripdrop_id = _normalize_platform_id(os.getenv("DRIPDROP_PLATFORM_ID", ""))
    if dripdrop_id:
        mapping[dripdrop_id] = ADAPTER_DRIPDROP_HMAC
    try:
        from app.network_config import active_networks, load_network_profile

        for net in active_networks():
            for entry in load_network_profile(net).platforms:
                platform_id = _normalize_platform_id(entry.platform_id)
                if platform_id:
                    mapping[platform_id] = entry.adapter
    except Exception:
        pass
    return mapping


def resolve_adapter(*, platform_id: str | None = None, url: str | None = None) -> str:
    """Pick an adapter. Unknown platform_id uses public HTTP unless the URL is DripDrop HLS."""
    mapped = _adapter_map().get(_normalize_platform_id(platform_id))
    if mapped and mapped != "*":
        return mapped
    client = get_dripdrop_video_client()
    if url and client.is_dripdrop_hls(url):
        return ADAPTER_DRIPDROP_HMAC
    wildcard = _adapter_map().get("*")
    return wildcard or ADAPTER_PUBLIC_HTTP


async def fetch_post_source(
    *,
    url: str,
    platform_id: str | None = None,
    post_object_id: str | None = None,
    creator_wallet_address: str | None = None,
    transaction_digest: str | None = None,
    event_sequence: int | None = None,
) -> str:
    """Return a fetchable URL (public or short-lived private lease)."""
    adapter = resolve_adapter(platform_id=platform_id, url=url)
    if adapter != ADAPTER_DRIPDROP_HMAC:
        return url

    client = get_dripdrop_video_client()
    asset_id = extract_asset_id_from_hls(url, client.media_host)
    if not asset_id or not client.enabled:
        return url
    if not (
        post_object_id
        and creator_wallet_address
        and transaction_digest
        and event_sequence is not None
    ):
        raise ValueError(
            "DripDrop HLS URL requires post_object_id, creator_wallet_address, "
            "transaction_digest, and event_sequence"
        )
    access = await client.source_access(
        asset_id=asset_id,
        post_object_id=post_object_id,
        creator_wallet_address=creator_wallet_address,
        hls_url=url,
        transaction_digest=transaction_digest,
        event_sequence=int(event_sequence),
    )
    source_url = access.get("sourceUrl")
    if not source_url:
        access = await client.source_access(
            asset_id=asset_id,
            post_object_id=post_object_id,
            creator_wallet_address=creator_wallet_address,
            hls_url=url,
            transaction_digest=transaction_digest,
            event_sequence=int(event_sequence),
        )
        source_url = access.get("sourceUrl")
    if not source_url:
        from app.services.analysis.media_fetcher import MediaNotReadyError

        raise MediaNotReadyError(url, message="source-access returned no URL")
    return str(source_url)


async def fetch_audio_source(
    *,
    content_commitment: str,
    resolution_request_id: str,
    platform_id: str | None = None,
    public_url: str | None = None,
) -> dict[str, Any]:
    """Lease or return a public URL for standalone catalog audio."""
    adapter = resolve_adapter(platform_id=platform_id, url=public_url)
    if adapter == ADAPTER_DRIPDROP_HMAC or (not public_url and get_dripdrop_video_client().enabled):
        client = get_dripdrop_video_client()
        if client.enabled:
            access = await client.audio_source_access(
                content_commitment=content_commitment,
                resolution_request_id=resolution_request_id,
            )
            source_url = str(access.get("sourceUrl") or "")
            if not source_url:
                from app.services.analysis.media_fetcher import MediaNotReadyError

                raise MediaNotReadyError("audio-source-lease", message="audio source lease missing URL")
            return access

    if public_url:
        return {"sourceUrl": public_url}

    from app.services.analysis.media_fetcher import MediaNotReadyError

    raise MediaNotReadyError("audio-source-lease", message="no audio source adapter or public URL")
