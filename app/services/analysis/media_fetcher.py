"""Download media from post URLs."""

from __future__ import annotations

import mimetypes
import os
import uuid
from pathlib import Path
from urllib.parse import urlparse

import httpx
import structlog

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


async def download_media(url: str, *, temp_dir: str | None = None) -> tuple[str, str]:
    base = temp_dir or os.getenv("TEMP_DIR", "/tmp/proof-of-creativity")
    Path(base).mkdir(parents=True, exist_ok=True)
    parsed = urlparse(url)
    ext = Path(parsed.path).suffix or mimetypes.guess_extension("application/octet-stream") or ".bin"
    dest = os.path.join(base, f"oracle_{uuid.uuid4().hex}{ext}")
    user_agent = os.getenv("POC_FETCH_USER_AGENT", os.getenv("DISCOVERY_FETCH_USER_AGENT", _DEFAULT_USER_AGENT))

    if url.startswith(("http://", "https://")):
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": user_agent})
            if resp.status_code in _MEDIA_NOT_READY_STATUSES:
                raise MediaNotReadyError(url, status_code=resp.status_code)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if exc.response is not None and exc.response.status_code in _MEDIA_NOT_READY_STATUSES:
                    raise MediaNotReadyError(url, status_code=exc.response.status_code) from exc
                raise
            Path(dest).write_bytes(resp.content)
    else:
        raise ValueError(f"Unsupported media URL scheme: {url}")

    content_type = mimetypes.guess_type(dest)[0] or "application/octet-stream"
    return dest, content_type
