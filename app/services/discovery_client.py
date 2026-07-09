"""HTTP client for myso-discovery-service lifecycle and provenance callbacks."""

from __future__ import annotations

import asyncio
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
        self.max_retries = int(os.getenv("DISCOVERY_CALLBACK_RETRIES", "3"))

    async def _post_with_retry(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> None:
        headers = {"Content-Type": "application/json"}
        if idempotency_key:
            headers["X-Idempotency-Key"] = idempotency_key
        delay = 0.5
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                return
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Discovery callback attempt failed",
                    url=url,
                    attempt=attempt + 1,
                    error=str(exc),
                )
                if attempt + 1 < self.max_retries:
                    await asyncio.sleep(delay)
                    delay *= 2
        logger.warning("Discovery callback failed after retries", url=url, error=str(last_exc))

    async def lifecycle_event(self, discovery_asset_id: str, event: str) -> None:
        if not self.enabled:
            return
        url = f"{self.base_url}/internal/lifecycle"
        payload = {"discovery_asset_id": discovery_asset_id, "event": event}
        await self._post_with_retry(
            url,
            payload,
            idempotency_key=f"{discovery_asset_id}:{event}",
        )

    async def record_provenance_hit(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        url = f"{self.base_url}/internal/provenance-hit"
        key = f"{payload.get('post_id')}:{payload.get('discovery_asset_id')}:{payload.get('decision')}"
        await self._post_with_retry(url, payload, idempotency_key=key)
