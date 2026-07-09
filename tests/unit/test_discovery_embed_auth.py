"""Unit tests for discovery embed auth and media_type validation."""

import pytest
from fastapi import HTTPException

from app.discovery.embedding_service import validate_embed_media_type, verify_embed_secret


def test_verify_embed_secret_debug_mode_without_secret():
    assert verify_embed_secret(None) is False or verify_embed_secret("Bearer x") is False


def test_validate_embed_media_type_accepts_poc_codes():
    assert validate_embed_media_type("image") == "image"
    assert validate_embed_media_type("VIDEO") == "video"
    assert validate_embed_media_type("audio") == "audio"
    assert validate_embed_media_type("2") == "video"
    assert validate_embed_media_type("image/jpeg") == "image"


def test_validate_embed_media_type_rejects_text():
    with pytest.raises(HTTPException) as exc:
        validate_embed_media_type("text/html")
    assert exc.value.status_code == 400

    with pytest.raises(HTTPException) as exc:
        validate_embed_media_type("application/json")
    assert exc.value.status_code == 400
