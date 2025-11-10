"""
WebSocket endpoints for real-time upload progress streaming
No authentication required, unlimited connections
"""
import json
import asyncio
import structlog
from fastapi import WebSocket, WebSocketDisconnect
from typing import Optional

logger = structlog.get_logger()

async def stream_upload_progress(
    websocket: WebSocket,
    upload_id: str,
    progress_tracker
):
    """
    Stream upload progress updates via WebSocket
    
    Args:
        websocket: FastAPI WebSocket connection
        upload_id: Upload identifier to track
        progress_tracker: ProgressTracker instance
    """
    await websocket.accept()
    
    logger.info("WebSocket connected for upload progress",
               upload_id=upload_id,
               client=websocket.client.host if websocket.client else "unknown")
    
    try:
        last_progress = None
        poll_interval = 0.5  # Poll every 500ms
        max_duration = 600  # 10 minutes max
        start_time = asyncio.get_event_loop().time()
        
        while True:
            # Check timeout
            if (asyncio.get_event_loop().time() - start_time) > max_duration:
                await websocket.send_json({
                    "type": "timeout",
                    "message": "Connection timed out after 10 minutes"
                })
                break
            
            # Get current progress
            current_progress = progress_tracker.get_progress(upload_id)
            
            if not current_progress:
                # Progress data not found yet, wait and retry
                await asyncio.sleep(poll_interval)
                continue
            
            # Only send if progress changed
            if current_progress != last_progress:
                # Send progress update
                await websocket.send_json({
                    "type": "progress",
                    "data": current_progress
                })
                
                last_progress = current_progress
                
                # Check if upload is complete or failed
                stage = current_progress.get("stage")
                if stage in ["complete", "failed", "pending_retry"]:
                    # Send final message
                    await websocket.send_json({
                        "type": "final",
                        "stage": stage,
                        "data": current_progress
                    })
                    
                    logger.info("Upload process finished",
                               upload_id=upload_id,
                               stage=stage,
                               progress=current_progress.get("progress_percent"))
                    break
            
            # Wait before next poll
            await asyncio.sleep(poll_interval)
        
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected",
                   upload_id=upload_id,
                   note="Background processing continues")
        # Client disconnected - processing continues in background
    
    except Exception as e:
        logger.error("WebSocket error",
                    upload_id=upload_id,
                    error=str(e))
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass  # Connection might be closed
    
    finally:
        try:
            await websocket.close()
        except:
            pass


async def handle_websocket_ping(websocket: WebSocket):
    """
    Simple ping/pong handler for connection health checks
    """
    await websocket.accept()
    
    try:
        while True:
            message = await websocket.receive_text()
            
            if message == "ping":
                await websocket.send_text("pong")
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": "Unknown command. Send 'ping' for health check."
                })
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("WebSocket ping error", error=str(e))
    finally:
        try:
            await websocket.close()
        except:
            pass

