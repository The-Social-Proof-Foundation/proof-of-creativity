"""
Background upload processing with progress streaming
Handles upload → analyze → blockchain submission pipeline
"""
import os
import time
import structlog
from pathlib import Path
from typing import List, Optional

from app.models.similarity import MediaMatch
from app.services.media_similarity import detect_image_similarity, detect_audio_similarity
from app.services.poc_video import analyze_video_similarity
from app.services.poc_video_types import VideoSimilarityAnalysis
from app.services.poc_submission import (
    attempt_proof_of_creativity_submission,
    attribution_type_from_chain_summary,
    build_upload_attribution_message,
    preview_poc_chain_summary_for_upload,
    preview_video_track_attestation,
)
from app.services.poc_utils import OFFCHAIN_DEFAULT_POC_CONFIG, poc_require_tx_when_post_id_from_env
from app.core.database import update_media_file_status, insert_attribution_record
from app.core.utils import cleanup_temp_file

logger = structlog.get_logger()


async def process_upload_background(
    upload_id: str,
    temp_file_path: str,
    filename: str,
    content_type: str,
    media_type: str,
    file_hash: str,
    post_id: Optional[str],
    spt_pool_id: Optional[str],
    force_reanalyze: bool,
    creator_address: Optional[str],
    upload_to_stream: bool,
    progress_tracker,
    storage_client,
    myso_client,
    royalty_free: bool = False,
):
    """Process upload with optional MySocial PoC submission + video track attestation."""
    _ = force_reanalyze

    try:
        start_time = time.time()
        logger.info(
            "Background upload processing started",
            upload_id=upload_id,
            media_type=media_type,
        )
        progress_tracker.update_progress(
            upload_id,
            stage="uploading",
            progress_percent=0,
            message="Starting upload to CDN...",
        )

        file_size = os.path.getsize(temp_file_path)

        def upload_progress_callback(bytes_uploaded: int):
            percent = min(int((bytes_uploaded / file_size) * 50), 50)
            progress_tracker.update_progress(
                upload_id,
                progress_percent=percent,
                message=f"Uploading to CDN... {percent}%",
            )

        ext = Path(filename).suffix or ".bin"
        simplified_filename = f"{upload_id}{ext}"

        with open(temp_file_path, "rb") as f:
            storage_uri = storage_client.upload(
                simplified_filename,
                f,
                media_type,
                progress_callback=upload_progress_callback,
            )

        streaming_uri = None
        if upload_to_stream and media_type == "video":
            try:
                with open(temp_file_path, "rb") as f:
                    streaming_uri = storage_client.upload_to_stream(
                        simplified_filename,
                        f,
                        media_type,
                        file_size,
                        metadata={
                            "media_id": upload_id,
                            "original_filename": filename,
                            "post_id": post_id or "",
                        },
                    )
            except Exception as e:
                logger.warning(
                    "Stream upload failed, continuing with primary storage",
                    upload_id=upload_id,
                    error=str(e),
                )

        progress_tracker.update_progress(
            upload_id,
            progress_percent=50,
            message="Upload complete, analyzing content...",
            storage_uri=storage_uri,
            streaming_uri=streaming_uri,
        )

        progress_tracker.update_progress(
            upload_id,
            stage="processing",
            progress_percent=55,
            message="Running AI similarity detection...",
        )

        video_analysis: Optional[VideoSimilarityAnalysis] = None
        matches: List[MediaMatch] = []
        if media_type == "image":
            matches = await detect_image_similarity(temp_file_path, upload_id)
        elif media_type == "audio":
            matches = await detect_audio_similarity(temp_file_path, upload_id)
        elif media_type == "video":
            video_analysis = analyze_video_similarity(temp_file_path, upload_id)
            matches = video_analysis.matches

        progress_tracker.update_progress(
            upload_id,
            progress_percent=80,
            message=f"Analysis complete, found {len(matches)} matches",
            matches_found=len(matches),
        )

        poc_cfg = OFFCHAIN_DEFAULT_POC_CONFIG
        if myso_client:
            try:
                poc_cfg = myso_client.get_poc_config()
            except Exception:
                pass
        media_type_code = {"image": 1, "audio": 3, "video": 2}.get(media_type, 1)

        summary = preview_poc_chain_summary_for_upload(
            media_cat=media_type,
            media_type_code=media_type_code,
            matches=matches,
            video_analysis=video_analysis,
            poc_config=poc_cfg,
            royalty_free=royalty_free,
        )
        video_att_dict = None
        if media_type == "video" and video_analysis is not None:
            vid_att = preview_video_track_attestation(video_analysis, poc_cfg)
            video_att_dict = vid_att.model_dump()

        tx_hash = None
        poc_submission_snap = None

        if post_id and myso_client:
            progress_tracker.update_progress(
                upload_id,
                stage="blockchain",
                progress_percent=85,
                message="Submitting to MySocial blockchain...",
            )
            try:
                res, vid_att, poc_submission_snap, summary = attempt_proof_of_creativity_submission(
                    myso_client=myso_client,
                    post_id=post_id,
                    media_cat=media_type,
                    media_type_code=media_type_code,
                    matches=matches,
                    video_analysis=video_analysis,
                    spt_pool_id=spt_pool_id,
                    royalty_free=royalty_free,
                )
                tx_hash = res.get("tx_hash")
                if poc_require_tx_when_post_id_from_env() and not tx_hash:
                    raise RuntimeError("MySocial PoC RPC returned no transaction digest (tx_hash)")
                video_att_dict = vid_att.model_dump() if vid_att else video_att_dict
                progress_tracker.update_progress(
                    upload_id,
                    progress_percent=95,
                    message="Blockchain submission complete",
                    tx_hash=tx_hash,
                    video_attestation=video_att_dict,
                )
            except Exception as e:
                err_msg = str(e)
                logger.error("Blockchain submission failed", upload_id=upload_id, error=err_msg)
                if poc_require_tx_when_post_id_from_env():
                    processing_time_fail = (time.time() - start_time) * 1000
                    fail_attribution_type = attribution_type_from_chain_summary(summary)
                    proof_fail = {
                        "file_hash": file_hash,
                        "original_filename": filename,
                        "upload_timestamp": time.time(),
                        "processing_time_ms": processing_time_fail,
                        "matches_found": len(matches),
                        "media_type": media_type,
                        "high_confidence_matches": len(
                            [m for m in matches if m.confidence_level == "high"]
                        ),
                        "match_types": list({m.match_type for m in matches}) if matches else [],
                        "max_similarity_score": max(
                            (m.similarity_score for m in matches), default=0.0
                        ),
                        "max_similarity_score_u64": summary.highest_similarity_score_u64,
                        "effective_threshold_u64": summary.effective_threshold_u64,
                        "poc_submission": None,
                        "poc_chain_summary": summary.model_dump(),
                        "royalty_free_requested": royalty_free,
                        "video_attestation": video_att_dict,
                        "post_id": post_id,
                        "spt_pool_id": spt_pool_id,
                        "poc_submission_error": err_msg,
                        "chain_submission_failed": True,
                    }
                    try:
                        insert_attribution_record(
                            media_id=upload_id,
                            attribution_type=fail_attribution_type,
                            proof_data=proof_fail,
                            tx_hash=None,
                        )
                    except Exception as att_e:
                        logger.warning(
                            "Could not record attribution after chain failure",
                            upload_id=upload_id,
                            error=str(att_e),
                        )
                    try:
                        update_media_file_status(
                            upload_id,
                            "completed",
                            storage_uri=storage_uri,
                            streaming_uri=streaming_uri,
                            processing_results={
                                "processing_time_ms": processing_time_fail,
                                "matches_found": len(matches),
                                "media_type": media_type,
                                "poc_submission_error": err_msg,
                                "chain_submission_failed": True,
                                "max_similarity_score_u64": summary.highest_similarity_score_u64,
                                "effective_threshold_u64": summary.effective_threshold_u64,
                                "poc_chain_summary": summary.model_dump(),
                            },
                        )
                    except Exception:
                        pass
                    completion_msg = (
                        "Oracle processing complete; MySocial PoC submission failed "
                        f"(required when post_id present): {err_msg}"
                    )
                    progress_tracker.mark_complete(
                        upload_id=upload_id,
                        storage_uri=storage_uri,
                        matches_found=len(matches),
                        attribution_type=fail_attribution_type,
                        tx_hash=None,
                        video_attestation=video_att_dict,
                        poc_submission=None,
                        poc_chain_summary=summary.model_dump(),
                        completion_message=completion_msg,
                        chain_submission_error=err_msg,
                    )
                    logger.warning(
                        "PoC chain submission failed but oracle pipeline completed",
                        upload_id=upload_id,
                        error=err_msg,
                    )
                    return
                summary = preview_poc_chain_summary_for_upload(
                    media_cat=media_type,
                    media_type_code=media_type_code,
                    matches=matches,
                    video_analysis=video_analysis,
                    poc_config=poc_cfg,
                    royalty_free=royalty_free,
                )
                progress_tracker.update_progress(
                    upload_id,
                    progress_percent=90,
                    message="Upload complete, blockchain submission pending retry",
                    error=f"Blockchain error: {err_msg}",
                )

        attribution_type = attribution_type_from_chain_summary(summary)

        processing_time = (time.time() - start_time) * 1000
        update_media_file_status(
            media_id=upload_id,
            status="completed",
            storage_uri=storage_uri,
            streaming_uri=streaming_uri,
            processing_results={
                "processing_time_ms": processing_time,
                "matches_found": len(matches),
                "media_type": media_type,
            },
        )

        proof_data = {
            "file_hash": file_hash,
            "original_filename": filename,
            "upload_timestamp": time.time(),
            "processing_time_ms": processing_time,
            "matches_found": len(matches),
            "media_type": media_type,
            "high_confidence_matches": len([m for m in matches if m.confidence_level == "high"]),
            "match_types": list({m.match_type for m in matches}) if matches else [],
            "max_similarity_score": max((m.similarity_score for m in matches), default=0.0),
            "max_similarity_score_u64": summary.highest_similarity_score_u64,
            "effective_threshold_u64": summary.effective_threshold_u64,
            "poc_submission": poc_submission_snap,
            "poc_chain_summary": summary.model_dump(),
            "royalty_free_requested": royalty_free,
            "video_attestation": video_att_dict,
            "post_id": post_id,
            "spt_pool_id": spt_pool_id,
        }

        insert_attribution_record(
            media_id=upload_id,
            attribution_type=attribution_type,
            proof_data=proof_data,
            tx_hash=tx_hash,
        )

        progress_tracker.mark_complete(
            upload_id=upload_id,
            storage_uri=storage_uri,
            matches_found=len(matches),
            attribution_type=attribution_type,
            tx_hash=tx_hash,
            video_attestation=video_att_dict,
            poc_submission=poc_submission_snap,
            poc_chain_summary=summary.model_dump(),
        )

    except Exception as e:
        logger.exception("Upload processing failed", upload_id=upload_id)
        progress_tracker.mark_failed(upload_id=upload_id, error=str(e), retry=True)
        try:
            update_media_file_status(upload_id, "failed")
        except Exception:
            pass
    finally:
        cleanup_temp_file(temp_file_path)
