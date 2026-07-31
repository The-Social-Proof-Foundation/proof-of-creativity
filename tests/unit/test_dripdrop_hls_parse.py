from app.services.dripdrop_video_client import extract_asset_id_from_hls


def test_extract_asset_id_from_clean_hls():
    url = "https://media.dripdrop.social/vid_abc123/master.m3u8"
    assert extract_asset_id_from_hls(url, "media.dripdrop.social") == "vid_abc123"


def test_reject_prod_prefix_and_query():
    assert (
        extract_asset_id_from_hls(
            "https://media.dripdrop.social/prod/vid_abc/master.m3u8",
            "media.dripdrop.social",
        )
        is None
    )
    assert (
        extract_asset_id_from_hls(
            "https://media.dripdrop.social/vid_abc/master.m3u8?x=1",
            "media.dripdrop.social",
        )
        is None
    )
