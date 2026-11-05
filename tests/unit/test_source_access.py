from app.services.source_access import (
    ADAPTER_DRIPDROP_HMAC,
    ADAPTER_PUBLIC_HTTP,
    _adapter_map,
    resolve_adapter,
)


def test_unknown_platform_uses_public_http(monkeypatch):
    _adapter_map.cache_clear()
    monkeypatch.delenv("DRIPDROP_PLATFORM_ID", raising=False)
    monkeypatch.delenv("POC_PLATFORM_ADAPTERS", raising=False)
    assert resolve_adapter(platform_id="0xabc", url="https://cdn.example/v.mp4") == ADAPTER_PUBLIC_HTTP


def test_dripdrop_platform_env(monkeypatch):
    _adapter_map.cache_clear()
    monkeypatch.setenv("DRIPDROP_PLATFORM_ID", "0x00dd")
    try:
        assert resolve_adapter(platform_id="0xdd", url="https://cdn.example/v.mp4") == ADAPTER_DRIPDROP_HMAC
    finally:
        _adapter_map.cache_clear()


def test_dripdrop_hls_host_fallback(monkeypatch):
    _adapter_map.cache_clear()
    monkeypatch.delenv("DRIPDROP_PLATFORM_ID", raising=False)
    monkeypatch.setenv("VIDEO_MEDIA_HOST", "media.dripdrop.social")
    url = "https://media.dripdrop.social/vid_abc123/master.m3u8"
    assert resolve_adapter(platform_id=None, url=url) == ADAPTER_DRIPDROP_HMAC
