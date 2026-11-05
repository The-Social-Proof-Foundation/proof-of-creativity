"""YAML source configuration loader for off-network discovery."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SourceEntry:
    url: str
    media_type: str = "image"
    title: str | None = None
    creator_x_handle: str | None = None
    trust_score: float | None = None


@dataclass
class SourceConfig:
    id: str
    adapter_type: str
    domain: str = "creative"
    content_kind: str = "media"
    trust_score: float = 0.5
    enabled: bool = True
    entries: list[SourceEntry] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


def default_sources_config_path() -> Path:
    explicit = os.getenv("DISCOVERY_SOURCES_CONFIG", "").strip()
    if explicit:
        return Path(explicit)
    return Path("config/discovery/sources.localnet.yaml")


def load_sources_config(path: Path | None = None) -> list[SourceConfig]:
    cfg_path = path or default_sources_config_path()
    if not cfg_path.is_file():
        return []
    with cfg_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or []
    if not isinstance(data, list):
        raise ValueError(f"discovery sources config must be a YAML list: {cfg_path}")
    sources: list[SourceConfig] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        entries = []
        for entry in item.get("entries") or []:
            if not isinstance(entry, dict) or not entry.get("url"):
                continue
            entries.append(
                SourceEntry(
                    url=str(entry["url"]),
                    media_type=str(entry.get("media_type") or "image"),
                    title=entry.get("title"),
                    creator_x_handle=entry.get("creator_x_handle"),
                    trust_score=entry.get("trust_score"),
                )
            )
        sources.append(
            SourceConfig(
                id=str(item.get("id") or item.get("adapter_type") or "unknown"),
                adapter_type=str(item.get("adapter_type") or "manual_curated"),
                domain=str(item.get("domain") or "creative"),
                content_kind=str(item.get("content_kind") or "media"),
                trust_score=float(item.get("trust_score") or 0.5),
                enabled=bool(item.get("enabled", True)),
                entries=entries,
                raw=item,
            )
        )
    return sources
