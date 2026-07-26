"""Tests for corpus embed media type validation."""

import pytest
from fastapi import HTTPException

from app.discovery.embedding_service import validate_embed_media_type


def test_validate_image():
    assert validate_embed_media_type("image") == "image"
    assert validate_embed_media_type("image/png") == "image"


def test_reject_text():
    with pytest.raises(HTTPException) as exc:
        validate_embed_media_type("application/json")
    assert exc.value.status_code == 400
