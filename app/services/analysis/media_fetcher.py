"""Download media from post URLs (public or dripdrop-backend signed source)."""

from __future__ import annotations

import mimetypes
import os
import uuid
from pathlib import Path
from urllib.parse import urlparse

import httpx
import structlog

from app.services.dripdrop_video_client import extract_asset_id_from_hls, get_dripdrop_video_client

logger = structlog.get_logger()

_DEFAULT_USER_AGENT = (
    "ProofOfCreativity-Discovery/1.0 (compatible; media-embed for provenance indexing)"
)

# HTTP statuses that mean the object is not ready yet (chain-first publish race).
_MEDIA_NOT_READY_STATUSES = {404, 408, 425, 429, 503}


class MediaNotReadyError(Exception):
    """Raised when media_urls cannot be fetched yet (e.g. R2 PUT still in flight)."""

    def __init__(self, url: str, status_code: int | None = None, message: str | None = None):
        self.url = url
        self.status_code = status_code
        detail = message or f"media not ready (HTTP {status_code})"
        super().__init__(f"{detail}: {url}")


async def download_media(
    url: str,
    *,
    temp_dir: str | None = None,
    post_object_id: str | None = None,
    creator_wallet_address: str | None = None,
    transaction_digest: str | None = None,
    event_sequence: int | None = None,
) -> tuple[str, str]:
    base = temp_dir or os.getenv("TEMP_DIR", "/tmp/proof-of-creativity")
    Path(base).mkdir(parents=True, exist_ok=True)

    client = get_dripdrop_video_client()
    asset_id = extract_asset_id_from_hls(url, client.media_host)
    if asset_id and client.enabled:
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
            # Replayed delivery returns no URL — mint again with new delivery via retry
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
            raise MediaNotReadyError(url, message="source-access returned no URL")
        return await _stream_download(source_url, base, preferred_ext=".bin")

    return await _stream_download(url, base)


async def _stream_download(url: str, base: str, preferred_ext: str | None = None) -> tuple[str, str]:
    parsed = urlparse(url)
    ext = preferred_ext or Path(parsed.path).suffix or mimetypes.guess_extension("application/octet-stream") or ".bin"
    # Prefer common video extensions when query-signed URLs have no path suffix
    if ext in (".bin", "") and "video" in (parsed.path or ""):
        ext = ".mp4"
    dest = os.path.join(base, f"oracle_{uuid.uuid4().hex}{ext}")
    user_agent = os.getenv("POC_FETCH_USER_AGENT", os.getenv("DISCOVERY_FETCH_USER_AGENT", _DEFAULT_USER_AGENT))

    if not url.startswith(("http://", "https://")):
        raise ValueError(f"Unsupported media URL scheme: {url}")

    max_bytes = int(os.getenv("VIDEO_MAX_SOURCE_BYTES", str(524288000)))
    async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
        async with client.stream("GET", url, headers={"User-Agent": user_agent}) as resp:
            if resp.status_code in _MEDIA_NOT_READY_STATUSES:
                raise MediaNotReadyError(url, status_code=resp.status_code)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if exc.response is not None and exc.response.status_code in _MEDIA_NOT_READY_STATUSES:
                    raise MediaNotReadyError(url, status_code=exc.response.status_code) from exc
                raise
            written = 0
            with open(dest, "wb") as fh:
                async for chunk in resp.aiter_bytes():
                    written += len(chunk)
                    if written > max_bytes:
                        fh.close()
                        Path(dest).unlink(missing_ok=True)
                        raise ValueError(f"Source exceeds max bytes ({max_bytes})")
                    fh.write(chunk)

    content_type = mimetypes.guess_type(dest)[0] or "application/octet-stream"
    return dest, content_type
