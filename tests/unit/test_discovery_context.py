"""Lifecycle FSM unit tests."""

from app.discovery.context import discovered_context, default_embedding_version


def test_discovered_context_metadata():
    ctx = discovered_context(
        discovery_asset_id="550e8400-e29b-41d4-a716-446655440000",
        creator_x_handle="artist",
        creator_confidence=0.9,
        identity_hash="0xabc",
    )
    meta = ctx.provenance_metadata()
    assert meta["corpus_scope"] == "discovered"
    assert meta["visibility"] == "internal"
    assert meta["identity_hash"] == "0xabc"
    assert ctx.embedding_version == default_embedding_version()
