"""Tests for manual_curated source adapter."""

import os
from unittest.mock import patch

from app.discovery.sources.config_loader import SourceConfig, SourceEntry
from app.discovery.sources.registry import discover_manual_curated


def test_manual_curated_requires_opt_in():
    source = SourceConfig(
        id="test",
        adapter_type="manual_curated",
        domain="creative",
        content_kind="media",
        enabled=True,
        entries=[SourceEntry(url="https://example.com/a.png", media_type="image")],
    )
    with patch.dict(os.environ, {"POC_USE_MANUAL_CURATED": "0"}, clear=False):
        assert discover_manual_curated(source) == []


def test_manual_curated_returns_media_records():
    source = SourceConfig(
        id="test",
        adapter_type="manual_curated",
        domain="creative",
        content_kind="media",
        enabled=True,
        entries=[SourceEntry(url="https://example.com/a.png", media_type="image", title="t")],
    )
    with patch.dict(os.environ, {"POC_USE_MANUAL_CURATED": "1"}, clear=False):
        records = discover_manual_curated(source)
    assert len(records) == 1
    assert records[0]["external_source_url"] == "https://example.com/a.png"
    assert records[0]["media_type"] == "image"
    assert records[0]["content_kind"] == "media"
