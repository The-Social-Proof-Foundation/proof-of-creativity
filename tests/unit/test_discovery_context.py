"""Tests for corpus embedding context."""

from app.discovery.context import discovered_context, default_embedding_version


def test_discovered_context_defaults():
    ctx = discovered_context(discovery_asset_id="asset-1")
    assert ctx.corpus_scope == "discovered"
    assert ctx.discovery_asset_id == "asset-1"
    assert ctx.embedding_version


def test_default_embedding_version_nonempty():
    assert default_embedding_version()
