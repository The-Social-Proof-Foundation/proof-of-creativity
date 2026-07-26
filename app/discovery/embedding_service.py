"""Embed off-network assets without persisting raw media."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import httpx
import structlog
from fastapi import HTTPException

from app.core.utils import cleanup_temp_file
from app.discovery.confidence import cold_start_work_confidence
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

_ALLOWED_MEDIA_TYPES = frozenset({"image", "audio", "video", "1", "2", "3"})


@dataclass
class EmbedResult:
    media_id: str
    work_confidence: float
    embedding_version: str
    embedding_model: str
    identity_hash: str | None = None


def _normalize_media_type(media_type: str) -> str:
    return media_type.strip().lower().split(";", 1)[0].strip()


def validate_embed_media_type(media_type: str) -> str:
    normalized = _normalize_media_type(media_type)
    if normalized.startswith("image/"):
        return "image"
    if normalized.startswith("video/"):
        return "video"
    if normalized.startswith("audio/"):
        return "audio"
    if normalized not in _ALLOWED_MEDIA_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                "media_type must be one of image, audio, video "
                f"(got {media_type!r}); factual text/JSON is not embeddable"
            ),
        )
    if normalized in ("1",):
        return "image"
    if normalized in ("2",):
        return "video"
    if normalized in ("3",):
        return "audio"
    return normalized


def _media_type_code(media_type: str) -> int:
    normalized = validate_embed_media_type(media_type)
    if normalized == "audio":
        return MEDIA_TYPE_AUDIO
    if normalized == "video":
        return MEDIA_TYPE_VIDEO
    return MEDIA_TYPE_IMAGE


def _is_unreadable_media_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    needles = (
        "cannot identify image file",
        "unidentifiedimageerror",
        "unsupported media",
        "invalid data found when processing input",
        "does not look like",
        "not a valid",
        "no such file",
        "failed to open",
    )
    return any(n in msg for n in needles)


async def embed_discovered_asset(
    *,
    discovery_asset_id: str,
    external_source_url: str,
    media_type: str,
    embedding_version: str | None = None,
    creator_x_handle: str | None = None,
    creator_confidence: float = 0.0,
    creator_candidate_id: str | None = None,
) -> EmbedResult:
    code = _media_type_code(media_type)
    identity_hash = resolve_identity_hash(creator_x_handle)
    ctx = discovered_context(
        discovery_asset_id=discovery_asset_id,
        embedding_version=embedding_version,
        creator_x_handle=creator_x_handle,
        creator_confidence=creator_confidence,
        identity_hash=identity_hash,
        creator_candidate_id=creator_candidate_id,
    )
    media_id = f"disc_{uuid.uuid4()}"
    try:
        path, _content_type = await download_media(external_source_url)
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        logger.warning(
            "Discovery embed media download failed",
            discovery_asset_id=discovery_asset_id,
            external_source_url=external_source_url,
            http_status=status,
            error=str(exc),
        )
        raise HTTPException(
            status_code=502,
            detail=f"failed to download media from {external_source_url}: HTTP {status}",
        ) from exc
    except httpx.HTTPError as exc:
        logger.warning(
            "Discovery embed media download failed",
            discovery_asset_id=discovery_asset_id,
            external_source_url=external_source_url,
            error=str(exc),
        )
        raise HTTPException(
            status_code=502,
            detail=f"failed to download media from {external_source_url}: {exc}",
        ) from exc
    try:
        set_active_embedding_context(ctx)
        matches = []
        try:
            if code == MEDIA_TYPE_IMAGE:
                matches = await detect_image_similarity(path, media_id)
            elif code == MEDIA_TYPE_AUDIO:
                matches = await detect_audio_similarity(path, media_id)
            elif code == MEDIA_TYPE_VIDEO:
                analysis = await analyze_video_similarity(path, media_id)
                matches = analysis.matches
            else:
                matches = await detect_image_similarity(path, media_id)
        except Exception as exc:
            if _is_unreadable_media_error(exc):
                logger.warning(
                    "Unsupported media for discovery embed",
                    discovery_asset_id=discovery_asset_id,
                    external_source_url=external_source_url,
                    media_type=media_type,
                    error=str(exc),
                )
                raise HTTPException(
                    status_code=422,
                    detail=f"unsupported media: downloaded bytes are not valid {media_type}",
                ) from exc
            logger.error(
                "Discovery embed similarity failed",
                discovery_asset_id=discovery_asset_id,
                external_source_url=external_source_url,
                media_type=media_type,
                error=str(exc),
            )
            raise

        work_confidence = max((m.similarity_score for m in matches), default=cold_start_work_confidence())
        ctx.work_confidence = work_confidence
        logger.info(
            "Discovery asset embedded",
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


# Backward-compatible alias during migration.
embed_discovered_asset = embed_discovered_asset
