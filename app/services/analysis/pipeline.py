"""Analysis pipeline wrapping existing media_similarity services."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field

import structlog

from app.core.database import lookup_mysocial_creator_for_media_ids
from app.core.utils import cleanup_temp_file, new_media_id
from app.db.discovery_repository import ProvenanceHitRepository
from app.db.oracle_repository import ConfigCacheRepository, MediaPostLinkRepository
from app.discovery.confidence import passes_off_network_thresholds
from app.discovery.store import DiscoveryStore
from app.models.similarity import MediaMatch
from app.services.analysis.media_fetcher import download_media
from app.services.analysis.scoring import similarity_float_to_u64_percent
from app.services.events import event_bus
from app.services.media_similarity import detect_audio_similarity, detect_image_similarity
from app.services.poc_utils import MEDIA_TYPE_AUDIO, MEDIA_TYPE_IMAGE, MEDIA_TYPE_VIDEO, OFFCHAIN_DEFAULT_POC_CONFIG
from app.services.poc_video import analyze_video_similarity

logger = structlog.get_logger()


def _select_discovered_match(matches: list[MediaMatch]) -> MediaMatch | None:
    discovered = [
        m
        for m in matches
        if (m.match_details or {}).get("corpus_scope") == "discovered"
    ]
    if not discovered:
        return None
    return max(discovered, key=lambda m: m.similarity_score)


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
    work_confidence: float = 0.0
    creator_confidence: float = 0.0
    discovery_asset_id: str | None = None
    matched_x_handle: str | None = None


class AnalysisService:
    def __init__(self) -> None:
        self.links = MediaPostLinkRepository()
        self.config_repo = ConfigCacheRepository()
        self.provenance = ProvenanceHitRepository()
        self.discovery = DiscoveryStore()

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
        *,
        creator_wallet_address: str | None = None,
        transaction_digest: str | None = None,
        event_sequence: int | None = None,
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
        path, _content_type = await download_media(
            media_url,
            post_object_id=post_id,
            creator_wallet_address=creator_wallet_address,
            transaction_digest=transaction_digest,
            event_sequence=event_sequence,
        )
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
                video_analysis = await asyncio.to_thread(
                    analyze_video_similarity, path, media_id
                )
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
            work_confidence = 0.0
            creator_confidence = 0.0
            discovery_asset_id = None
            matched_x_handle = None

            discovered_match = _select_discovered_match(matches)
            details = (discovered_match.match_details or {}) if discovered_match else {}
            if discovered_match and details:
                work_confidence = float(details.get("work_confidence") or discovered_match.similarity_score)
                creator_confidence = float(details.get("creator_confidence") or 0.0)
                discovery_asset_id = details.get("discovery_asset_id")
                matched_x_handle = details.get("creator_x_handle")
                identity_hash = details.get("identity_hash")

            creator_candidate_id = details.get("creator_candidate_id")

            if highest > 0 and not creator:
                if passes_off_network_thresholds(
                    identity_hash=identity_hash,
                    creator_confidence=creator_confidence,
                    work_confidence=work_confidence,
                ):
                    off_network = True
                    needs_review = False
                    decision = "redirect_escrow"
                elif discovered_match:
                    needs_review = True
                    decision = "needs_review"
                elif matches:
                    needs_review = True
                    decision = "needs_review"
                else:
                    decision = "below_threshold"

                if discovered_match:
                    self.provenance.record(
                        network=network,
                        post_id=post_id,
                        query_media_id=media_id,
                        discovery_asset_id=str(discovery_asset_id) if discovery_asset_id else None,
                        creator_candidate_id=str(creator_candidate_id) if creator_candidate_id else None,
                        similarity_score=float(discovered_match.similarity_score),
                        match_type=str(discovered_match.match_type),
                        work_confidence=work_confidence,
                        creator_confidence=creator_confidence,
                        decision=decision,
                    )
                    if discovery_asset_id and off_network:
                        self.discovery.transition_asset(str(discovery_asset_id), "match_detected")
                    if needs_review and discovery_asset_id:
                        self.provenance.record(
                            network=network,
                            post_id=post_id,
                            query_media_id=media_id,
                            discovery_asset_id=str(discovery_asset_id),
                            creator_candidate_id=str(creator_candidate_id) if creator_candidate_id else None,
                            similarity_score=float(discovered_match.similarity_score),
                            match_type=str(discovered_match.match_type),
                            work_confidence=work_confidence,
                            creator_confidence=creator_confidence,
                            decision="needs_review",
                        )

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
                work_confidence=work_confidence,
                creator_confidence=creator_confidence,
                discovery_asset_id=str(discovery_asset_id) if discovery_asset_id else None,
                matched_x_handle=matched_x_handle,
            )
            if needs_review:
                await event_bus.publish(
                    "post.analysis.needs_review",
                    {
                        "network": network,
                        "post_id": post_id,
                        "reason": "unresolvable_creator",
                        "discovery_asset_id": discovery_asset_id,
                        "work_confidence": work_confidence,
                        "creator_confidence": creator_confidence,
                    },
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
