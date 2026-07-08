"""PoC decision engine mirroring Move contract logic."""

from __future__ import annotations

import os
from dataclasses import dataclass

from app.db.oracle_repository import ConfigCacheRepository
from app.network_config import get_settings
from app.services.analysis.pipeline import AnalysisResult
from app.services.poc_submission import build_image_or_audio_submission
from app.services.poc_utils import (
    DERIVATIVE_TARGET_ESCROW,
    DERIVATIVE_TARGET_WALLET,
    MEDIA_TYPE_VIDEO,
    OFFCHAIN_DEFAULT_POC_CONFIG,
)

DEFAULT_CONFIG = dict(OFFCHAIN_DEFAULT_POC_CONFIG)

SELF_MATCH_REASONING_SUFFIX = (
    " Self-match: poster equals matched creator; treating as original (no redirect/vault)."
)


def normalize_address(addr: str | None) -> str | None:
    if not addr:
        return None
    normalized = str(addr).strip().lower()
    if normalized.startswith("0x"):
        normalized = normalized[2:]
    return normalized or None


def addresses_equal(a: str | None, b: str | None) -> bool:
    left = normalize_address(a)
    right = normalize_address(b)
    return left is not None and right is not None and left == right


def post_owner(post: dict) -> str | None:
    owner = post.get("creator_address") or post.get("owner")
    if owner:
        return str(owner).strip()
    return None


def apply_self_match_short_circuit(
    *,
    derivative: bool,
    creator: str | None,
    post_owner_addr: str | None,
    reasoning: str,
) -> tuple[bool, str | None, str]:
    if derivative and creator and post_owner_addr and addresses_equal(creator, post_owner_addr):
        return False, None, reasoning + SELF_MATCH_REASONING_SUFFIX
    return derivative, creator, reasoning


@dataclass
class PoCSubmission:
    post_id: str
    media_type: int
    highest_similarity_score: int
    original_creator: str | None
    derivative_redirection_target: int
    embedded_audio_only_derivative: bool
    apply_explicit_outcome: bool
    explicit_poc_outcome: int
    reasoning: str
    evidence_urls: list[str]
    off_network: bool = False
    identity_hash: str | None = None
    needs_review: bool = False


class DecisionEngine:
    def __init__(self) -> None:
        self.config_repo = ConfigCacheRepository()
        self.settings = get_settings()

    def _load_config(self, network: str) -> dict:
        try:
            return self.config_repo.get(network) or DEFAULT_CONFIG
        except Exception:
            return DEFAULT_CONFIG

    def build_submission(self, post: dict, analysis: AnalysisResult) -> PoCSubmission:
        cfg = self._load_config(analysis.network)
        if analysis.needs_review:
            return PoCSubmission(
                post_id=analysis.post_id,
                media_type=analysis.media_type,
                highest_similarity_score=analysis.highest_similarity_u64,
                original_creator=None,
                derivative_redirection_target=DERIVATIVE_TARGET_WALLET,
                embedded_audio_only_derivative=analysis.embedded_audio_only_derivative,
                apply_explicit_outcome=False,
                explicit_poc_outcome=0,
                reasoning=analysis.reasoning + " Needs manual review.",
                evidence_urls=[],
                needs_review=True,
            )

        image_thr = int(cfg.get("image_threshold") or 95)
        video_thr = int(cfg.get("video_threshold") or 95)
        audio_thr = int(cfg.get("audio_threshold") or 95)

        def resolve_creator(media_ids: list[str]) -> str | None:
            if analysis.off_network and analysis.identity_hash:
                return None
            return analysis.original_creator

        if analysis.media_type == MEDIA_TYPE_VIDEO:
            score = analysis.highest_similarity_u64
            creator = analysis.original_creator
            derivative = score >= video_thr and creator is not None
            reasoning = analysis.reasoning
        else:
            score, derivative, creator, reasoning = build_image_or_audio_submission(
                analysis.matches,
                analysis.media_type,
                image_thr,
                video_thr,
                audio_thr,
                resolve_creator,
            )

        off_network_derivative = False
        if (
            analysis.off_network
            and analysis.identity_hash
            and analysis.creator_confidence
            >= float(os.getenv("DISCOVERY_X_HANDLE_CONFIDENCE_THRESHOLD", "0.85"))
            and score
            >= (
            audio_thr if analysis.media_type == 3 else video_thr if analysis.media_type == 2 else image_thr
            )
        ):
            off_network_derivative = True
            derivative = True

        derivative, creator, reasoning = apply_self_match_short_circuit(
            derivative=derivative,
            creator=creator,
            post_owner_addr=post_owner(post),
            reasoning=reasoning,
        )

        redirect = DERIVATIVE_TARGET_WALLET
        if derivative:
            if off_network_derivative or (analysis.off_network and self.settings.off_network_force_escrow):
                redirect = DERIVATIVE_TARGET_ESCROW
            else:
                redirect = DERIVATIVE_TARGET_WALLET

        return PoCSubmission(
            post_id=analysis.post_id,
            media_type=analysis.media_type,
            highest_similarity_score=score,
            original_creator=creator if derivative else None,
            derivative_redirection_target=redirect,
            embedded_audio_only_derivative=analysis.embedded_audio_only_derivative,
            apply_explicit_outcome=False,
            explicit_poc_outcome=0,
            reasoning=reasoning,
            evidence_urls=[],
            off_network=analysis.off_network,
            identity_hash=analysis.identity_hash,
        )
