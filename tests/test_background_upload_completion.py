"""Background upload completion path (streaming follow-up processing)."""

import pytest

try:
    import torch  # noqa: F401 — background pipeline imports embedding stack
except ImportError:
    pytest.skip("torch required for background_tasks tests", allow_module_level=True)

from unittest.mock import MagicMock, patch

from app.services.background_tasks import process_upload_background
from app.services.poc_video_types import VideoSimilarityAnalysis


@pytest.fixture
def temp_video_path(tmp_path):
    p = tmp_path / "tiny.mp4"
    p.write_bytes(b"\x00" * 64)
    return str(p)


@pytest.mark.asyncio
async def test_process_upload_background_video_reaches_mark_complete(temp_video_path):
    """Ensures completion path runs (start_time defined; mark_complete accepts video_attestation)."""
    progress = MagicMock()
    progress.update_progress = MagicMock(return_value=True)
    progress.mark_complete = MagicMock(return_value=True)
    storage = MagicMock()
    storage.upload = MagicMock(return_value="mock://storage/object")

    empty_analysis = VideoSimilarityAnalysis(matches=[])

    with (
        patch("app.services.background_tasks.update_media_file_status"),
        patch("app.services.background_tasks.insert_attribution_record"),
        patch("app.services.background_tasks.cleanup_temp_file"),
        patch(
            "app.services.background_tasks.analyze_video_similarity",
            return_value=empty_analysis,
        ),
    ):
        await process_upload_background(
            upload_id="test-upload-completion",
            temp_file_path=temp_video_path,
            filename="clip.mp4",
            content_type="video/mp4",
            media_type="video",
            file_hash="0" * 64,
            post_id=None,
            spt_pool_id=None,
            force_reanalyze=False,
            creator_address=None,
            upload_to_stream=False,
            progress_tracker=progress,
            storage_client=storage,
            myso_client=None,
            royalty_free=False,
        )

    progress.mark_complete.assert_called_once()
    kw = progress.mark_complete.call_args.kwargs
    assert kw["storage_uri"] == "mock://storage/object"
    assert kw["matches_found"] == 0
    assert "video_attestation" in kw


@pytest.mark.asyncio
async def test_strict_tx_required_chain_failure_completes_oracle(temp_video_path):
    """RPC/PoC failure must not mark_failed when MYSO_POC_REQUIRE_TX_WHEN_POST_ID is set."""
    progress = MagicMock()
    progress.update_progress = MagicMock(return_value=True)
    progress.mark_complete = MagicMock(return_value=True)
    progress.mark_failed = MagicMock(return_value=True)
    storage = MagicMock()
    storage.upload = MagicMock(return_value="mock://storage/object")

    empty_analysis = VideoSimilarityAnalysis(matches=[])
    myso = MagicMock()

    def boom(*args, **kwargs):
        raise RuntimeError("connection refused")

    with (
        patch("app.services.background_tasks.update_media_file_status") as mock_status,
        patch("app.services.background_tasks.insert_attribution_record"),
        patch("app.services.background_tasks.cleanup_temp_file"),
        patch(
            "app.services.background_tasks.analyze_video_similarity",
            return_value=empty_analysis,
        ),
        patch(
            "app.services.background_tasks.attempt_proof_of_creativity_submission",
            side_effect=boom,
        ),
        patch(
            "app.services.background_tasks.poc_require_tx_when_post_id_from_env",
            return_value=True,
        ),
    ):
        await process_upload_background(
            upload_id="test-upload-chain-fail",
            temp_file_path=temp_video_path,
            filename="clip.mp4",
            content_type="video/mp4",
            media_type="video",
            file_hash="0" * 64,
            post_id="0xabc",
            spt_pool_id=None,
            force_reanalyze=False,
            creator_address=None,
            upload_to_stream=False,
            progress_tracker=progress,
            storage_client=storage,
            myso_client=myso,
            royalty_free=False,
        )

    progress.mark_failed.assert_not_called()
    progress.mark_complete.assert_called_once()
    mc_kw = progress.mark_complete.call_args.kwargs
    assert mc_kw["tx_hash"] is None
    assert mc_kw["chain_submission_error"] == "connection refused"
    mock_status.assert_called_once()
    assert mock_status.call_args[0][1] == "completed"


def test_mark_complete_serializes_optional_video_attestation():
    """ProgressTracker forwards video_attestation into Redis payload via update_progress."""
    from app.core.progress_tracker import ProgressTracker

    fake_redis = MagicMock()
    fake_redis.enabled = True
    fake_redis.get.return_value = (
        '{"upload_id":"x","predicted_url":"http://example","progress_percent":0,'
        '"stage":"processing","matches_found":0,"attribution_type":null}'
    )
    fake_redis.set = MagicMock(return_value=True)

    pt = ProgressTracker(fake_redis)
    assert pt.mark_complete(
        upload_id="x",
        storage_uri="mock://done",
        matches_found=0,
        attribution_type="original",
        video_attestation={"embedded_audio_only_derivative": False},
    )

    wrote = fake_redis.set.call_args[0][1]
    assert "video_attestation" in wrote
    assert "embedded_audio_only_derivative" in wrote
