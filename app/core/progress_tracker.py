"""
Redis-based progress tracking for streaming uploads
Ephemeral state management for real-time WebSocket updates
"""
import json
import time
import structlog
from typing import Optional, Dict, Any
from enum import Enum

logger = structlog.get_logger()

class UploadStage(str, Enum):
    """Upload processing stages"""
    VALIDATING = "validating"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    BLOCKCHAIN = "blockchain"
    COMPLETE = "complete"
    FAILED = "failed"
    PENDING_RETRY = "pending_retry"

class ProgressTracker:
    """Track upload progress in Redis for WebSocket streaming"""
    
    def __init__(self, redis_client):
        """
        Initialize progress tracker
        
        Args:
            redis_client: RedisCache instance
        """
        self.redis = redis_client
        self.ttl = 3600  # 1 hour TTL for progress data
    
    def create_upload(self, upload_id: str, predicted_url: str, post_id: Optional[str] = None) -> bool:
        """
        Create initial upload progress entry
        
        Args:
            upload_id: Unique upload identifier (media_id)
            predicted_url: Pre-generated CDN URL
            post_id: MySocial post object ID (optional)
        """
        if not self.redis or not self.redis.enabled:
            return False
        
        try:
            progress_data = {
                "upload_id": upload_id,
                "predicted_url": predicted_url,
                "post_id": post_id,
                "stage": UploadStage.VALIDATING,
                "progress_percent": 0,
                "message": "Validating upload...",
                "created_at": time.time(),
                "updated_at": time.time(),
                "error": None,
                "blockchain_tx_hash": None,
                "matches_found": 0,
                "attribution_type": None,
            }
            
            key = f"upload_progress:{upload_id}"
            self.redis.set(key, json.dumps(progress_data), ttl=self.ttl)
            
            logger.debug("Upload progress created", upload_id=upload_id)
            return True
            
        except Exception as e:
            logger.warning("Failed to create upload progress", error=str(e))
            return False
    
    def update_progress(
        self,
        upload_id: str,
        stage: Optional[UploadStage] = None,
        progress_percent: Optional[int] = None,
        message: Optional[str] = None,
        error: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Update upload progress
        
        Args:
            upload_id: Upload identifier
            stage: Current processing stage
            progress_percent: Progress percentage (0-100)
            message: Status message
            error: Error message if failed
            **kwargs: Additional fields (blockchain_tx_hash, matches_found, etc.)
        """
        if not self.redis or not self.redis.enabled:
            return False
        
        try:
            key = f"upload_progress:{upload_id}"
            
            # Get current progress
            current_data = self.redis.get(key)
            if not current_data:
                logger.warning("Upload progress not found", upload_id=upload_id)
                return False
            
            progress_data = json.loads(current_data)
            
            # Update fields
            if stage is not None:
                progress_data["stage"] = stage
            if progress_percent is not None:
                progress_data["progress_percent"] = progress_percent
            if message is not None:
                progress_data["message"] = message
            if error is not None:
                progress_data["error"] = error
                progress_data["stage"] = UploadStage.FAILED
            
            # Update any additional fields
            progress_data.update(kwargs)
            progress_data["updated_at"] = time.time()
            
            # Save back to Redis
            self.redis.set(key, json.dumps(progress_data), ttl=self.ttl)
            
            logger.debug("Upload progress updated",
                        upload_id=upload_id,
                        stage=stage,
                        progress=progress_percent)
            return True
            
        except Exception as e:
            logger.warning("Failed to update upload progress", error=str(e))
            return False
    
    def get_progress(self, upload_id: str) -> Optional[Dict[str, Any]]:
        """Get current upload progress"""
        if not self.redis or not self.redis.enabled:
            return None
        
        try:
            key = f"upload_progress:{upload_id}"
            data = self.redis.get(key)
            
            if not data:
                return None
            
            return json.loads(data)
            
        except Exception as e:
            logger.warning("Failed to get upload progress", error=str(e))
            return None
    
    def mark_complete(
        self,
        upload_id: str,
        storage_uri: str,
        matches_found: int,
        attribution_type: str,
        blockchain_tx_hash: Optional[str] = None
    ) -> bool:
        """Mark upload as complete with final results"""
        return self.update_progress(
            upload_id=upload_id,
            stage=UploadStage.COMPLETE,
            progress_percent=100,
            message="Upload complete!",
            storage_uri=storage_uri,
            matches_found=matches_found,
            attribution_type=attribution_type,
            blockchain_tx_hash=blockchain_tx_hash
        )
    
    def mark_failed(self, upload_id: str, error: str, retry: bool = True) -> bool:
        """Mark upload as failed"""
        stage = UploadStage.PENDING_RETRY if retry else UploadStage.FAILED
        return self.update_progress(
            upload_id=upload_id,
            stage=stage,
            message=f"Upload failed: {error}",
            error=error
        )
    
    def delete_progress(self, upload_id: str) -> bool:
        """Delete progress data (cleanup)"""
        if not self.redis or not self.redis.enabled:
            return False
        
        try:
            key = f"upload_progress:{upload_id}"
            self.redis.delete(key)
            return True
        except Exception as e:
            logger.warning("Failed to delete upload progress", error=str(e))
            return False

