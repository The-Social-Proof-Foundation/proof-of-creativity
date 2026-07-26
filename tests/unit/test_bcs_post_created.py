"""Unit tests for PostCreatedEvent BCS decoder."""

from __future__ import annotations

import pytest

from app.chain.bcs_post_created import BcsDecodeError, decode_post_created_event


def _uleb128(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            break
    return bytes(out)


def _addr(byte_val: int) -> bytes:
    raw = bytearray(32)
    raw[-1] = byte_val
    return bytes(raw)


def _string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return _uleb128(len(encoded)) + encoded


def _option_none() -> bytes:
    return b"\x00"


def _option_some_vec_string(items: list[str]) -> bytes:
    body = _uleb128(len(items)) + b"".join(_string(item) for item in items)
    return b"\x01" + body


def _post_access_public() -> bytes:
    return _uleb128(0)


def encode_post_created_current(
    *,
    media_urls: list[str] | None = None,
    poc_redirection_kind: int = 0,
    with_organization: bool = True,
) -> bytes:
    """Current Move PostCreatedEvent layout (PostAccess, no enable_poc/spot)."""
    parts = [
        _addr(1),  # post_id
        _addr(2),  # owner
        _addr(3),  # profile_id
        _addr(4),  # platform_id
        b"\x00",  # permissions
        _string("hello"),
        _string("post"),
        _option_none(),  # parent_post_id
        _option_none(),  # mentions
        _option_some_vec_string(media_urls) if media_urls else _option_none(),
        _option_none(),  # metadata_json
        _post_access_public(),
        _option_none(),  # promotion_id
        _option_none(),  # revenue_redirect_to
        _option_none(),  # revenue_redirect_percentage
        b"\x00",  # enable_spt
        _option_none(),  # spt_id
        bytes([poc_redirection_kind]),
        _addr(2),  # actor_address
        _option_none(),  # sub_agent_id
    ]
    if with_organization:
        parts.append(_option_none())  # organization_id
    parts.append(b"\x00")  # action_identity_class
    return b"".join(parts)


def encode_post_created_legacy(*, enable_poc: bool = True) -> bytes:
    """Pre-PostAccess layout kept as decoder fallback."""
    parts = [
        _addr(1),
        _addr(2),
        _addr(3),
        _addr(4),
        b"\x00",
        _string("legacy"),
        _string("post"),
        _option_none(),
        _option_none(),
        _option_some_vec_string(["https://example.com/legacy.jpg"]),
        _option_none(),
        _option_none(),  # mydata_id
        _option_none(),
        _option_none(),
        _option_none(),
        b"\x00",  # enable_spt
        b"\x01" if enable_poc else b"\x00",
        b"\x00",  # enable_spot
        _option_none(),  # spot_id
        _option_none(),  # spt_id
    ]
    return b"".join(parts)


def test_decode_post_created_current_with_organization():
    raw = encode_post_created_current(
        media_urls=["https://example.com/a.jpg", "https://example.com/b.mp4"],
        poc_redirection_kind=0,
    )
    decoded = decode_post_created_event(raw)
    assert decoded.post_id.endswith("01")
    assert decoded.owner.endswith("02")
    assert decoded.content == "hello"
    assert decoded.post_access_kind == "public"
    assert decoded.mydata_id is None
    assert decoded.poc_redirection_kind == 0
    assert decoded.media_urls == ["https://example.com/a.jpg", "https://example.com/b.mp4"]
    assert decoded.enable_poc is None

    payload = decoded.to_payload()
    assert payload["creator"] == payload["owner"]
    assert payload["media_urls"] == decoded.media_urls
    assert "enable_poc" not in payload
    assert payload["post_access_kind"] == "public"
    assert payload["poc_redirection_kind"] == 0


def test_decode_post_created_legacy_layout():
    raw = encode_post_created_legacy(enable_poc=True)
    decoded = decode_post_created_event(raw)
    assert decoded.content == "legacy"
    assert decoded.poc_redirection_kind == 0
    assert decoded.media_urls == ["https://example.com/legacy.jpg"]
    assert decoded.enable_poc is True
    assert decoded.to_payload()["enable_poc"] is True


def test_decode_empty_contents_raises():
    with pytest.raises(BcsDecodeError):
        decode_post_created_event(b"")


def test_decode_invalid_bytes_raises():
    with pytest.raises(BcsDecodeError):
        decode_post_created_event(b"\x01\x02\x03")
