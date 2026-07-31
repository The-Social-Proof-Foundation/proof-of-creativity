"""HMAC client for dripdrop-backend video source-access / analysis-complete."""

from __future__ import annotations

import hashlib
import hmac
import os
import time
import uuid
from typing import Any
from urllib.parse import urlparse

import httpx
import structlog

logger = structlog.get_logger()


def _parse_key_ring(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in (raw or "").split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        kid, secret = part.split(":", 1)
        kid, secret = kid.strip(), secret.strip()
        if kid and secret:
            out[kid] = secret
    return out


def extract_asset_id_from_hls(url: str, media_host: str) -> str | None:
    try:
        parsed = urlparse(url)
    except Exception:
        return None
    if parsed.scheme != "https":
        return None
    host = (media_host or "").replace("https://", "").replace("http://", "").strip("/")
    if parsed.hostname != host:
        return None
    if parsed.query or parsed.fragment:
        return None
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) != 2 or parts[1] != "master.m3u8":
        return None
    asset_id = parts[0]
    if not asset_id.startswith("vid_"):
        return None
    return asset_id


class DripdropVideoClient:
    def __init__(self) -> None:
        self.base_url = (os.getenv("DRIPDROP_BACKEND_URL") or "").rstrip("/")
        self.media_host = (
            (os.getenv("VIDEO_MEDIA_HOST") or "media.dripdrop.social")
            .replace("https://", "")
            .replace("http://", "")
            .strip("/")
        )
        keys = _parse_key_ring(os.getenv("VIDEO_POC_HMAC_KEYS") or "")
        self._kid = next(iter(keys.keys()), "")
        self._secret = keys.get(self._kid, "")
        self.enabled = bool(self.base_url and self._kid and self._secret)

    def is_dripdrop_hls(self, url: str) -> bool:
        return extract_asset_id_from_hls(url, self.media_host) is not None

    def _sign(self, raw_body: str) -> dict[str, str]:
        ts = str(int(time.time()))
        sig = hmac.new(
            self._secret.encode("utf-8"),
            f"{ts}.{raw_body}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return {
            "X-Timestamp": ts,
            "X-Signature": sig,
            "X-Key-Id": self._kid,
            "Content-Type": "application/json",
        }

    async def source_access(
        self,
        *,
        asset_id: str,
        post_object_id: str,
        creator_wallet_address: str,
        hls_url: str,
        transaction_digest: str,
        event_sequence: int,
    ) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("Dripdrop video client is not configured")
        delivery_id = f"dlv_{uuid.uuid4().hex}"
        body = {
            "deliveryId": delivery_id,
            "postObjectId": post_object_id,
            "creatorWalletAddress": creator_wallet_address,
            "hlsUrl": hls_url,
            "transactionDigest": transaction_digest,
            "eventSequence": event_sequence,
        }
        import json

        raw = json.dumps(body, separators=(",", ":"))
        headers = self._sign(raw)
        url = f"{self.base_url}/v1/internal/video-assets/{asset_id}/source-access"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, content=raw, headers=headers)
            if resp.status_code >= 400:
                logger.warning(
                    "source-access failed",
                    status=resp.status_code,
                    # never log signed URL bodies
                    body=resp.text[:500],
                )
            resp.raise_for_status()
            return resp.json()

    async def analysis_complete(
        self,
        *,
        asset_id: str,
        post_object_id: str,
        creator_wallet_address: str,
        hls_url: str,
        transaction_digest: str,
        event_sequence: int,
        status: str,
    ) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("Dripdrop video client is not configured")
        import json

        body = {
            "postObjectId": post_object_id,
            "creatorWalletAddress": creator_wallet_address,
            "hlsUrl": hls_url,
            "transactionDigest": transaction_digest,
            "eventSequence": event_sequence,
            "status": status,
        }
        raw = json.dumps(body, separators=(",", ":"))
        headers = self._sign(raw)
        url = f"{self.base_url}/v1/internal/video-assets/{asset_id}/analysis-complete"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, content=raw, headers=headers)
            resp.raise_for_status()
            return resp.json()


_client: DripdropVideoClient | None = None


def get_dripdrop_video_client() -> DripdropVideoClient:
    global _client
    if _client is None:
        _client = DripdropVideoClient()
    return _client
