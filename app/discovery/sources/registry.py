"""Source adapters for off-network creative media discovery."""

from __future__ import annotations

import os

from app.discovery.sources.config_loader import SourceConfig, SourceEntry


def manual_curated_enabled() -> bool:
    return os.getenv("POC_USE_MANUAL_CURATED", os.getenv("DISCOVERY_USE_MANUAL_CURATED", "")).lower() in (
        "1",
        "true",
        "yes",
    )


def discover_manual_curated(config: SourceConfig) -> list[dict]:
    if config.adapter_type != "manual_curated" or not config.enabled:
        return []
    if not manual_curated_enabled():
        return []
    records: list[dict] = []
    for entry in config.entries:
        media_type = _normalize_media_type(entry.media_type)
        records.append(
            {
                "external_source_url": entry.url,
                "media_type": media_type,
                "content_kind": "media",
                "title": entry.title,
                "creator_x_handle": entry.creator_x_handle,
                "trust_score": entry.trust_score if entry.trust_score is not None else config.trust_score,
                "metadata": {
                    "title": entry.title,
                    "source": "manual_curated",
                },
            }
        )
    return records


def _normalize_media_type(media_type: str) -> str:
    normalized = media_type.strip().lower()
    if normalized.startswith("image/"):
        return "image"
    if normalized.startswith("video/"):
        return "video"
    if normalized.startswith("audio/"):
        return "audio"
    if normalized in ("1",):
        return "image"
    if normalized in ("2",):
        return "video"
    if normalized in ("3",):
        return "audio"
    return normalized


def discover_source(config: SourceConfig) -> list[dict]:
    if config.domain != "creative" or config.content_kind != "media":
        return []
    if config.adapter_type == "manual_curated":
        return discover_manual_curated(config)
    return []
