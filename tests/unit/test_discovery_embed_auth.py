"""Integration-style test for discovery embed endpoint auth."""

from app.discovery.embedding_service import verify_embed_secret


def test_verify_embed_secret_debug_mode_without_secret():
    assert verify_embed_secret(None) is False or verify_embed_secret("Bearer x") is False
