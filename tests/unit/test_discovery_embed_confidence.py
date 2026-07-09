"""Unit tests for embed cold-start confidence behavior."""

import os
from unittest.mock import AsyncMock, patch

import pytest

from app.discovery.embedding_service import embed_discovered_asset


@pytest.mark.asyncio
async def test_embed_cold_start_returns_zero_work_confidence():
    with patch.dict(os.environ, {"DISCOVERY_COLD_START_WORK_CONFIDENCE": "0.0"}, clear=False):
        with patch(
            "app.discovery.embedding_service.download_media",
            new=AsyncMock(return_value=("/tmp/fake.jpg", "image/jpeg")),
        ):
            with patch(
                "app.discovery.embedding_service.detect_image_similarity",
                new=AsyncMock(return_value=[]),
            ):
                with patch("app.discovery.embedding_service.cleanup_temp_file"):
                    result = await embed_discovered_asset(
                        discovery_asset_id="550e8400-e29b-41d4-a716-446655440000",
                        external_source_url="https://example.com/img.jpg",
                        media_type="image",
                        creator_x_handle="creator",
                        creator_confidence=0.9,
                    )
    assert result.work_confidence == 0.0
