"""Embed discovered external assets without persisting raw media."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

import structlog

from app.core.utils import cleanup_temp_file, new_media_id
from app.discovery.context import discovered_context
from app.discovery.identity import resolve_identity_hash
from app.services.analysis.media_fetcher import download_media
from app.services.media_similarity import (
    detect_audio_similarity,
    detect_image_similarity,
    set_active_embedding_context,
)
from app.services.poc_utils import MEDIA_TYPE_AUDIO, MEDIA_TYPE_IMAGE, MEDIA_TYPE_VIDEO
from app.services.poc_video import analyze_video_similarity

logger = structlog.get_logger()


@dataclass
class EmbedResult:
    media_id: str
    work_confidence: float
    embedding_version: str
    embedding_model: str
    identity_hash: str | None = None


def _media_type_code(media_type: str) -> int:
    normalized = media_type.strip().lower()
    if normalized in ("audio", "3"):
        return MEDIA_TYPE_AUDIO
    if normalized in ("video", "2"):
        return MEDIA_TYPE_VIDEO
    return MEDIA_TYPE_IMAGE


async def embed_discovered_asset(
    *,
    discovery_asset_id: str,
    external_source_url: str,
    media_type: str,
    embedding_version: str | None = None,
    creator_x_handle: str | None = None,
    creator_confidence: float = 0.0,
) -> EmbedResult:
    identity_hash = resolve_identity_hash(creator_x_handle)
    ctx = discovered_context(
        discovery_asset_id=discovery_asset_id,
        embedding_version=embedding_version,
        creator_x_handle=creator_x_handle,
        creator_confidence=creator_confidence,
        identity_hash=identity_hash,
    )
    media_id = f"disc_{uuid.uuid4()}"
    path, _content_type = await download_media(external_source_url)
    try:
        set_active_embedding_context(ctx)
        code = _media_type_code(media_type)
        matches = []
        if code == MEDIA_TYPE_IMAGE:
            matches = await detect_image_similarity(path, media_id)
        elif code == MEDIA_TYPE_AUDIO:
            matches = await detect_audio_similarity(path, media_id)
        elif code == MEDIA_TYPE_VIDEO:
            analysis = await analyze_video_similarity(path, media_id)
            matches = analysis.matches
        else:
            matches = await detect_image_similarity(path, media_id)

        work_confidence = max((m.similarity_score for m in matches), default=0.95)
        ctx.work_confidence = work_confidence
        logger.info(
            "Discovered asset embedded",
            discovery_asset_id=discovery_asset_id,
            media_id=media_id,
            work_confidence=work_confidence,
        )
        return EmbedResult(
            media_id=media_id,
            work_confidence=work_confidence,
            embedding_version=ctx.embedding_version,
            embedding_model=ctx.embedding_model,
            identity_hash=identity_hash,
        )
    finally:
        set_active_embedding_context(None)
        cleanup_temp_file(path)


def verify_embed_secret(authorization: str | None) -> bool:
    secret = os.getenv("DISCOVERY_EMBED_SECRET", "").strip()
    if not secret:
        return os.getenv("DEBUG", "").lower() in ("1", "true", "yes")
    if not authorization:
        return False
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        return False
    return authorization[len(prefix) :].strip() == secret
