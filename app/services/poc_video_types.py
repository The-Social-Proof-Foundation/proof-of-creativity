"""Pure video PoC types + chain branching (no torch / video_processing imports)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from app.models.similarity import MediaMatch
from app.services.poc_utils import similarity_float_to_u64_percent


@dataclass
class VideoSimilarityAnalysis:
    """Structured video analysis output (visual vs embedded audio tracks)."""

    matches: List[MediaMatch]
    max_visual_similarity: float = 0.0
    best_visual_match_media_id: Optional[str] = None
    audio_fingerprint_match: bool = False
    embedded_audio_similarity: float = 0.0
    best_audio_match_media_id: Optional[str] = None


@dataclass
class VideoPoCChainDecision:
    embedded_audio_only_derivative: bool
    highest_similarity_score_u64: int
    original_creator: Optional[str]
    reasoning_summary: str


def decide_video_poc_for_chain(
    analysis: VideoSimilarityAnalysis,
    video_threshold_pct: int,
    audio_threshold_pct: int,
    visual_creator: Optional[str],
    audio_creator: Optional[str],
) -> VideoPoCChainDecision:
    """
    Decide Move args for VIDEO posts.

    Thresholds from chain PoCConfig are integer percents (0–100). Float similarities use
    the same rounding as other modalities via similarity_float_to_u64_percent.
    """
    visual_pct = similarity_float_to_u64_percent(analysis.max_visual_similarity)
    audio_pct = similarity_float_to_u64_percent(analysis.embedded_audio_similarity)

    visual_deriv = visual_pct >= video_threshold_pct and bool(visual_creator)
    audio_deriv = audio_pct >= audio_threshold_pct and bool(audio_creator)

    if visual_deriv:
        embedded = False
        score_u64 = visual_pct
        creator = visual_creator
        reason = f"Video visual match {score_u64}/100 (>={video_threshold_pct}); redirection target creator."
        if audio_deriv:
            reason += f" Embedded audio match {audio_pct}/100 also detected; chain path uses full video derivative."
        return VideoPoCChainDecision(embedded, score_u64, creator, reason)

    if audio_deriv:
        embedded = True
        return VideoPoCChainDecision(
            True,
            audio_pct,
            audio_creator,
            f"Embedded-audio-only derivative {audio_pct}/100 (>={audio_threshold_pct}).",
        )

    embedded = False
    return VideoPoCChainDecision(
        False,
        visual_pct if visual_pct >= audio_pct else audio_pct,
        None,
        "No derivative above thresholds with attributable creators.",
    )
