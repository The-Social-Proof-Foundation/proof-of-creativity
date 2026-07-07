"""Analysis pipeline wrapping existing media_similarity services."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field

import structlog

from app.core.database import lookup_mysocial_creator_for_media_ids
from app.core.utils import cleanup_temp_file, new_media_id
from app.db.oracle_repository import ConfigCacheRepository, MediaPostLinkRepository
from app.models.similarity import MediaMatch
from app.services.analysis.media_fetcher import download_media
from app.services.analysis.scoring import similarity_float_to_u64_percent
from app.services.events import event_bus
from app.services.media_similarity import detect_audio_similarity, detect_image_similarity
from app.services.poc_utils import MEDIA_TYPE_AUDIO, MEDIA_TYPE_IMAGE, MEDIA_TYPE_VIDEO, OFFCHAIN_DEFAULT_POC_CONFIG
from app.services.poc_video import analyze_video_similarity

logger = structlog.get_logger()


@dataclass
class AnalysisResult:
    network: str
    post_id: str
    media_url: str
    media_index: int
    media_type: int
    media_id: str
    matches: list[MediaMatch] = field(default_factory=list)
    highest_similarity_u64: int = 0
    original_creator: str | None = None
    identity_hash: str | None = None
    off_network: bool = False
    needs_review: bool = False
    reasoning: str = ""
    embedded_audio_only_derivative: bool = False


class AnalysisService:
    def __init__(self) -> None:
        self.links = MediaPostLinkRepository()
        self.config_repo = ConfigCacheRepository()

    def _load_thresholds(self, network: str) -> tuple[int, int, int]:
        try:
            cfg = self.config_repo.get(network) or OFFCHAIN_DEFAULT_POC_CONFIG
        except Exception:
            cfg = OFFCHAIN_DEFAULT_POC_CONFIG
        return (
            int(cfg.get("image_threshold") or 95),
            int(cfg.get("video_threshold") or 95),
            int(cfg.get("audio_threshold") or 95),
        )

    async def analyze(
        self,
        network: str,
        post_id: str,
        media_url: str,
        media_index: int,
        media_type: int,
    ) -> AnalysisResult:
        await event_bus.publish(
            "post.analysis.progress",
            {
                "network": network,
                "post_id": post_id,
                "stage": "download",
                "pct": 10,
            },
        )
        path, _content_type = await download_media(media_url)
        media_id = new_media_id()
        try:
            await event_bus.publish(
                "post.analysis.progress",
                {
                    "network": network,
                    "post_id": post_id,
                    "stage": "fingerprint",
                    "pct": 40,
                },
            )
            matches: list[MediaMatch] = []
            embedded_audio_only = False
            if media_type == MEDIA_TYPE_IMAGE:
                matches = await detect_image_similarity(path, media_id)
            elif media_type == MEDIA_TYPE_AUDIO:
                matches = await detect_audio_similarity(path, media_id)
            elif media_type == MEDIA_TYPE_VIDEO:
                video_analysis = await analyze_video_similarity(path, media_id)
                matches = video_analysis.matches
                from app.services.poc_video_types import decide_video_poc_for_chain

                _img_thr, video_thr, audio_thr = self._load_thresholds(network)
                dec = decide_video_poc_for_chain(
                    video_analysis, video_thr, audio_thr, None, None
                )
                embedded_audio_only = dec.embedded_audio_only_derivative
            else:
                matches = await detect_image_similarity(path, media_id)

            self.links.link(network, post_id, media_id, media_url=media_url, media_index=media_index)

            highest = similarity_float_to_u64_percent(
                max((m.similarity_score for m in matches), default=0.0)
            )
            creator = lookup_mysocial_creator_for_media_ids([m.media_id for m in matches])
            identity_hash = None
            off_network = False
            needs_review = False
            if highest > 0 and not creator:
                payload_match = next(iter(matches), None)
                if payload_match and payload_match.match_details:
                    identity_hash = payload_match.match_details.get("identity_hash")
                    off_network = bool(identity_hash)
                    if off_network:
                        needs_review = False
                    else:
                        needs_review = True

            reasoning = f"Analyzed {media_url}; matches={len(matches)}; top_score_u64={highest}."
            result = AnalysisResult(
                network=network,
                post_id=post_id,
                media_url=media_url,
                media_index=media_index,
                media_type=media_type,
                media_id=media_id,
                matches=matches,
                highest_similarity_u64=highest,
                original_creator=creator,
                identity_hash=identity_hash,
                off_network=off_network,
                needs_review=needs_review,
                reasoning=reasoning,
                embedded_audio_only_derivative=embedded_audio_only,
            )
            if needs_review:
                await event_bus.publish(
                    "post.analysis.needs_review",
                    {"network": network, "post_id": post_id, "reason": "unresolvable_creator"},
                )
            await event_bus.publish(
                "post.analysis.complete",
                {
                    "network": network,
                    "post_id": post_id,
                    "outcome": "pending_decision",
                    "score": highest,
                    "similarity_detected": highest > 0,
                },
            )
            return result
        finally:
            cleanup_temp_file(path)
