"""Smoke tests for AnalysisService video path (sync analyze_video_similarity)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.analysis.pipeline import AnalysisService
from app.services.poc_utils import MEDIA_TYPE_VIDEO
from app.services.poc_video_types import VideoSimilarityAnalysis


@pytest.mark.asyncio
async def test_analyze_video_uses_to_thread_not_bare_await():
    """Regression: awaiting sync analyze_video_similarity raised TypeError after work finished."""
    service = AnalysisService()
    service.links = MagicMock()
    service.config_repo = MagicMock()
    service.config_repo.get.return_value = {
        "image_threshold": 95,
        "video_threshold": 95,
        "audio_threshold": 95,
    }
    service.provenance = MagicMock()
    service.discovery = MagicMock()

    video_result = VideoSimilarityAnalysis(
        matches=[],
        max_visual_similarity=0.0,
        audio_fingerprint_match=False,
        embedded_audio_similarity=0.0,
    )

    with (
        patch(
            "app.services.analysis.pipeline.download_media",
            new_callable=AsyncMock,
            return_value=("/tmp/fake.mp4", "video/mp4"),
        ),
        patch(
            "app.services.analysis.pipeline.analyze_video_similarity",
            return_value=video_result,
        ) as mock_video,
        patch(
            "app.services.analysis.pipeline.event_bus.publish",
            new_callable=AsyncMock,
        ),
        patch(
            "app.services.analysis.pipeline.lookup_mysocial_creator_for_media_ids",
            return_value=None,
        ),
        patch("app.services.analysis.pipeline.cleanup_temp_file"),
        patch(
            "app.services.analysis.pipeline.new_media_id",
            return_value="media-video-1",
        ),
    ):
        result = await service.analyze(
            network="localnet",
            post_id="0xabc",
            media_url="https://example.com/v.mp4",
            media_index=0,
            media_type=MEDIA_TYPE_VIDEO,
        )

    mock_video.assert_called_once_with("/tmp/fake.mp4", "media-video-1")
    assert result.media_id == "media-video-1"
    assert result.matches == []
    assert result.post_id == "0xabc"
