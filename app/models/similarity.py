"""
Pydantic models for similarity matching and response data structures.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

from .media import ConfidenceLevel

class MatchType(str, Enum):
    """Enumeration of match types."""
    EMBEDDING = "embedding"
    FINGERPRINT = "fingerprint"
    PERCEPTUAL_HASH = "perceptual_hash"
    EXACT = "exact"

class MediaMatch(BaseModel):
    """Model for a similarity match between media files."""
    media_id: str = Field(..., description="ID of the matched media")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score (0.0 to 1.0)")
    match_type: MatchType = Field(..., description="Type of match")
    confidence_level: ConfidenceLevel = Field(..., description="Confidence level")
    match_details: Optional[Dict[str, Any]] = Field(default=None, description="Additional match details")
    
    class Config:
        use_enum_values = True

class UploadResponse(BaseModel):
    """Response model for media upload and processing."""
    media_id: str = Field(..., description="Unique identifier for uploaded media")
    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME type of the uploaded file")
    file_size: int = Field(..., description="File size in bytes")
    file_hash: str = Field(..., description="SHA-256 hash of file content")
    storage_uri: str = Field(..., description="Storage URI (GCS or Walrus)")
    matches: List[MediaMatch] = Field(default=[], description="Similar media found")
    processing_status: str = Field(..., description="Processing status")
    message: str = Field(..., description="Human-readable message")
    
class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Overall service status")
    version: str = Field(..., description="API version")
    components: Dict[str, Any] = Field(..., description="Component health status") 