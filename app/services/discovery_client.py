"""HTTP client for myso-discovery-service lifecycle and provenance callbacks."""

from __future__ import annotations

import os
from typing import Any

import httpx
import structlog

logger = structlog.get_logger()


class DiscoveryClient:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or os.getenv("DISCOVERY_SERVICE_URL", "")).rstrip("/")
        self.enabled = bool(self.base_url) and os.getenv("DISCOVERY_ENABLED", "").lower() in (
            "1",
            "true",
            "yes",
        )

    async def lifecycle_event(self, discovery_asset_id: str, event: str) -> None:
        if not self.enabled:
            return
        url = f"{self.base_url}/internal/lifecycle"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    url,
                    json={"discovery_asset_id": discovery_asset_id, "event": event},
                )
                resp.raise_for_status()
        except Exception as exc:
            logger.warning("Discovery lifecycle callback failed", error=str(exc))

    async def record_provenance_hit(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        url = f"{self.base_url}/internal/provenance-hit"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
        except Exception as exc:
            logger.warning("Discovery provenance hit callback failed", error=str(exc))
