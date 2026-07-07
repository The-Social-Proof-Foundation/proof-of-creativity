"""Download media from post URLs."""

from __future__ import annotations

import mimetypes
import os
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlparse

import httpx
import structlog

logger = structlog.get_logger()


async def download_media(url: str, *, temp_dir: str | None = None) -> tuple[str, str]:
    base = temp_dir or os.getenv("TEMP_DIR", "/tmp/proof-of-creativity")
    Path(base).mkdir(parents=True, exist_ok=True)
    parsed = urlparse(url)
    ext = Path(parsed.path).suffix or mimetypes.guess_extension("application/octet-stream") or ".bin"
    dest = os.path.join(base, f"oracle_{uuid.uuid4().hex}{ext}")

    if url.startswith(("http://", "https://")):
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            Path(dest).write_bytes(resp.content)
    else:
        raise ValueError(f"Unsupported media URL scheme: {url}")

    content_type = mimetypes.guess_type(dest)[0] or "application/octet-stream"
    return dest, content_type
