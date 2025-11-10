"""
Background upload processing with progress streaming
Handles upload → analyze → blockchain submission pipeline
"""
import os
import structlog
from typing import Optional, Callable
from pathlib import Path

logger = structlog.get_logger()

async def process_upload_background(
    upload_id: str,
    temp_file_path: str,
    filename: str,
    content_type: str,
    media_type: str,
    file_hash: str,
    post_id: Optional[str],
    upload_to_stream: bool,
    progress_tracker,
    storage_client,
    mys_client,
    detect_similarity_func: Callable,
):
    """
    Background task for processing upload with streaming progress
    
    Args:
        upload_id: Media ID for tracking
        temp_file_path: Path to temporary file
        filename: Original filename
        content_type: MIME type
        media_type: Category (image/audio/video)
        file_hash: SHA-256 hash
        post_id: MySocial post ID (optional)
        upload_to_stream: Whether to upload video to Cloudflare Stream (opt-in)
        progress_tracker: ProgressTracker instance
        storage_client: StorageClient instance
        mys_client: MySocialClient instance (optional)
        detect_similarity_func: Function to detect similarity
    """
    from app.core.database import (
        insert_media_file, update_media_file_status,
        insert_attribution_record
    )
    from app.core.utils import cleanup_temp_file
    import time
    
    start_time = time.time()
    
    try:
        # Stage 1: Upload to R2 (0-50%)
        progress_tracker.update_progress(
            upload_id,
            stage="uploading",
            progress_percent=0,
            message="Starting upload to CDN..."
        )
        
        # Get file size for progress
        file_size = os.path.getsize(temp_file_path)
        
        # Upload with progress callback
        def upload_progress_callback(bytes_uploaded: int):
            """Callback for upload progress"""
            percent = min(int((bytes_uploaded / file_size) * 50), 50)  # 0-50%
            progress_tracker.update_progress(
                upload_id,
                progress_percent=percent,
                message=f"Uploading to CDN... {percent}%"
            )
        
        # Simplified filename: just media_id + extension
        ext = Path(filename).suffix or ".bin"
        simplified_filename = f"{upload_id}{ext}"
        
        # Upload to R2 (for processing and backup)
        with open(temp_file_path, "rb") as f:
            storage_uri = storage_client.upload(
                simplified_filename,
                f,
                media_type,
                progress_callback=upload_progress_callback
            )
        
        # Upload to Cloudflare Stream (for video streaming) - opt-in per request
        streaming_uri = None
        if upload_to_stream and media_type == "video":
            try:
                logger.info("Uploading to Cloudflare Stream (opt-in enabled)",
                           upload_id=upload_id)
                
                with open(temp_file_path, "rb") as f:
                    streaming_uri = storage_client.upload_to_stream(
                        simplified_filename,
                        f,
                        media_type,
                        file_size,
                        metadata={
                            "media_id": upload_id,
                            "original_filename": filename,
                            "post_id": post_id if post_id else ""
                        }
                    )
                
                if streaming_uri:
                    logger.info("Video uploaded to Stream successfully", 
                               upload_id=upload_id,
                               streaming_uri=streaming_uri)
                else:
                    logger.warning("Stream upload returned no URI",
                                  upload_id=upload_id)
            except Exception as e:
                logger.warning("Stream upload failed, continuing with R2 only",
                              upload_id=upload_id,
                              error=str(e))
        elif upload_to_stream and media_type != "video":
            logger.debug("Stream upload requested but media is not video, skipping",
                        upload_id=upload_id,
                        media_type=media_type)
        
        progress_tracker.update_progress(
            upload_id,
            progress_percent=50,
            message="Upload complete, analyzing content...",
            storage_uri=storage_uri,
            streaming_uri=streaming_uri
        )
        
        # Insert media file record
        insert_media_file(
            media_id=upload_id,
            filename=simplified_filename,
            original_filename=filename,
            content_type=content_type,
            file_size=file_size,
            file_hash=file_hash,
            status="processing"
        )
        
        # Stage 2: AI Analysis (50-80%)
        progress_tracker.update_progress(
            upload_id,
            stage="processing",
            progress_percent=55,
            message="Running AI similarity detection..."
        )
        
        # Run similarity detection
        matches = await detect_similarity_func(temp_file_path, upload_id, media_type)
        
        progress_tracker.update_progress(
            upload_id,
            progress_percent=80,
            message=f"Analysis complete, found {len(matches)} matches",
            matches_found=len(matches)
        )
        
        # Determine attribution type
        attribution_type = _determine_attribution_type(matches)
        
        # Stage 3: Blockchain Submission (80-95%)
        blockchain_tx_hash = None
        
        # PRINT STATEMENT FOR JANE - Check if we'll submit
        print(f"\n[POC] Blockchain submission check - post_id={post_id}, mys_client={'initialized' if mys_client else 'None'}\n")
        
        if post_id and mys_client:
            try:
                progress_tracker.update_progress(
                    upload_id,
                    stage="blockchain",
                    progress_percent=85,
                    message="Submitting to MySocial blockchain..."
                )
                
                # Map media type
                media_type_code = {"image": 1, "audio": 3, "video": 2}.get(media_type, 1)
                
                # Calculate highest similarity
                highest_similarity = int(max([m.similarity_score for m in matches]) * 100) if matches else 0
                
                # Get original creator (if derivative)
                original_creator = None
                if matches and len([m for m in matches if m.confidence_level == "high"]) > 0:
                    # TODO: Would need MySocial address mapping
                    pass
                
                # Submit to blockchain
                mys_result = mys_client.submit_poc_analysis(
                    post_id=post_id,
                    media_type=media_type_code,
                    similarity_score=highest_similarity,
                    original_creator=original_creator
                )
                
                blockchain_tx_hash = mys_result.get("tx_hash")
                
                progress_tracker.update_progress(
                    upload_id,
                    progress_percent=95,
                    message="Blockchain submission complete",
                    blockchain_tx_hash=blockchain_tx_hash
                )
                
                logger.info("PoC submitted to MySocial",
                           upload_id=upload_id,
                           post_id=post_id,
                           tx_hash=blockchain_tx_hash)
                
            except Exception as e:
                logger.error("Blockchain submission failed",
                            upload_id=upload_id,
                            error=str(e))
                # Mark as pending retry but don't fail the whole upload
                progress_tracker.update_progress(
                    upload_id,
                    progress_percent=90,
                    message="Upload complete, blockchain submission pending retry",
                    error=f"Blockchain error: {str(e)}"
                )
        
        # Update media file status
        processing_time = (time.time() - start_time) * 1000
        update_media_file_status(
            media_id=upload_id,
            status="completed",
            storage_uri=storage_uri,
            streaming_uri=streaming_uri,
            processing_results={
                "processing_time_ms": processing_time,
                "matches_found": len(matches),
                "media_type": media_type
            }
        )
        
        # Record attribution
        proof_data = {
            "file_hash": file_hash,
            "original_filename": filename,
            "upload_timestamp": time.time(),
            "processing_time_ms": processing_time,
            "matches_found": len(matches),
            "media_type": media_type,
            "high_confidence_matches": len([m for m in matches if m.confidence_level == "high"]),
            "match_types": list(set(m.match_type for m in matches)) if matches else [],
            "max_similarity_score": max([m.similarity_score for m in matches]) if matches else 0.0
        }
        
        insert_attribution_record(
            media_id=upload_id,
            attribution_type=attribution_type,
            proof_data=proof_data,
            blockchain_tx_hash=blockchain_tx_hash
        )
        
        # Mark complete (100%)
        progress_tracker.mark_complete(
            upload_id=upload_id,
            storage_uri=storage_uri,
            matches_found=len(matches),
            attribution_type=attribution_type,
            blockchain_tx_hash=blockchain_tx_hash
        )
        
        logger.info("Upload processing complete",
                   upload_id=upload_id,
                   matches=len(matches),
                   attribution=attribution_type,
                   processing_time_ms=processing_time)
        
    except Exception as e:
        logger.error("Upload processing failed",
                    upload_id=upload_id,
                    error=str(e))
        
        # Mark as failed/pending retry
        progress_tracker.mark_failed(
            upload_id=upload_id,
            error=str(e),
            retry=True  # Mark as pending retry
        )
        
        # Update database
        try:
            update_media_file_status(upload_id, "failed")
        except:
            pass
    
    finally:
        # Cleanup temp file
        cleanup_temp_file(temp_file_path)


def _determine_attribution_type(matches) -> str:
    """Determine attribution type based on matches"""
    if not matches:
        return "original"
    
    high_confidence_matches = [m for m in matches if m.confidence_level == "high"]
    
    if len(high_confidence_matches) == 0:
        return "original"
    elif len(high_confidence_matches) <= 2:
        return "derivative"
    else:
        return "remix"

