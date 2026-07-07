"""Unit tests for oracle event parser."""

from app.chain.event_parser import infer_media_type, parse_post_created


def test_parse_post_created_maps_owner_to_creator():
    parsed = parse_post_created(
        {
            "post_id": "0xabc",
            "owner": "0xowner",
            "enable_poc": True,
            "media_urls": ["https://x/y.jpg"],
        }
    )
    assert parsed is not None
    assert parsed["creator"] == "0xowner"
    assert parsed["owner"] == "0xowner"


def test_parse_post_created_handles_option_empty_media_urls():
    parsed = parse_post_created(
        {
            "post_id": "0xabc",
            "owner": "0xowner",
            "enable_poc": True,
            "media_urls": None,
        }
    )
    assert parsed is not None
    assert parsed["media_urls"] == []


def test_parse_post_created_extracts_media_urls():
    parsed = parse_post_created(
        {
            "post_id": "0xabc",
            "creator": "0xcreator",
            "enable_poc": True,
            "media_urls": ["https://x/y.jpg"],
            "media_types": [1],
        }
    )
    assert parsed is not None
    assert parsed["post_id"] == "0xabc"
    assert parsed["media_urls"] == ["https://x/y.jpg"]
    assert parsed["enable_poc"] is True


def test_parse_skips_when_no_post_id():
    assert parse_post_created({"enable_poc": True}) is None


def test_infer_media_type_from_extension():
    assert infer_media_type("https://x/a.mp4", 0, []) == 2
    assert infer_media_type("https://x/a.mp3", 0, []) == 3
    assert infer_media_type("https://x/a.png", 0, [2]) == 2
