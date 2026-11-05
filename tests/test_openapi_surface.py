"""OpenAPI must describe the oracle-only HTTP surface (no leftover upload routes)."""

from app.main import app


def test_openapi_has_oracle_not_upload():
    spec = app.openapi()
    paths = spec["paths"]
    assert "/oracle/health" in paths
    assert "/oracle/assets/{asset_id}" in paths
    assert "/oracle/posts/{post_id}" in paths
    assert "/health" in paths
    for path in paths:
        assert not path.startswith("/upload")
        assert "/uploads" not in path
        assert path != "/media/{media_id}/attribution"
    description = spec["info"]["description"].lower()
    assert "mediaasset" in description
    assert "/v1/videos/uploads" not in description
