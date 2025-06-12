"""
Pydantic models for media-related data structures.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum

class MediaType(str, Enum):
    """Enumeration of supported media types."""
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"

class ProcessingStatus(str, Enum):
    """Enumeration of processing statuses."""
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class ConfidenceLevel(str, Enum):
    """Enumeration of confidence levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class MediaFileBase(BaseModel):
    """Base model for media file information."""
    filename: str = Field(..., description="Name of the uploaded file")
    original_filename: str = Field(..., description="Original name of the file")
    content_type: str = Field(..., description="MIME type of the file")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    file_hash: str = Field(..., description="SHA-256 hash of file content")

class MediaFileCreate(MediaFileBase):
    """Model for creating a new media file record."""
    upload_user_id: Optional[str] = Field(None, description="ID of the uploading user")
    upload_ip: Optional[str] = Field(None, description="IP address of uploader")

class MediaFile(MediaFileBase):
    """Complete media file model with all fields."""
    media_id: str = Field(..., description="Unique identifier for the media")
    storage_uri: Optional[str] = Field(None, description="Storage URI (GCS/Walrus)")
    upload_user_id: Optional[str] = Field(None, description="ID of the uploading user")
    upload_ip: Optional[str] = Field(None, description="IP address of uploader")
    status: ProcessingStatus = Field(default=ProcessingStatus.PROCESSING)
    processing_results: Optional[Dict[str, Any]] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        use_enum_values = True

class EmbeddingBase(BaseModel):
    """Base model for embeddings."""
    media_id: str = Field(..., description="ID of the associated media")
    kind: str = Field(..., description="Type of embedding (image, video_frame, etc.)")
    embedding: List[float] = Field(..., description="Embedding vector")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Embedding(EmbeddingBase):
    """Complete embedding model."""
    id: Optional[str] = Field(None, description="Unique embedding ID")
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class FingerprintBase(BaseModel):
    """Base model for audio fingerprints."""
    fp_hash: str = Field(..., description="Fingerprint hash")
    media_id: str = Field(..., description="ID of the associated media")
    offset_seconds: float = Field(default=0.0, description="Time offset in seconds")

class Fingerprint(FingerprintBase):
    """Complete fingerprint model."""
    id: Optional[str] = Field(None, description="Unique fingerprint ID")
    fingerprint_data: Optional[bytes] = Field(None, description="Raw fingerprint data")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class AttributionRecordBase(BaseModel):
    """Base model for blockchain attribution records."""
    media_id: str = Field(..., description="ID of the associated media")
    attribution_type: str = Field(default="original", description="Type of attribution")
    
class AttributionRecord(AttributionRecordBase):
    """Complete attribution record model."""
    id: Optional[str] = Field(None, description="Unique attribution ID")
    blockchain_tx_hash: Optional[str] = Field(None, description="Blockchain transaction hash")
    blockchain_address: Optional[str] = Field(None, description="Blockchain address")
    proof_data: Optional[Dict[str, Any]] = Field(default=None, description="Proof data")
    created_at: datetime = Field(default_factory=datetime.utcnow)

class MediaDimensions(BaseModel):
    """Model for media dimensions and properties."""
    width: Optional[int] = Field(None, description="Width in pixels")
    height: Optional[int] = Field(None, description="Height in pixels")
    duration_seconds: Optional[float] = Field(None, description="Duration in seconds")
    fps: Optional[float] = Field(None, description="Frames per second")
    sample_rate: Optional[int] = Field(None, description="Audio sample rate")
    channels: Optional[int] = Field(None, description="Number of audio channels")
    format: Optional[str] = Field(None, description="Media format")
    
class MediaUploadRequest(BaseModel):
    """Model for media upload request parameters."""
    extract_keyframes: bool = Field(default=True, description="Extract keyframes from video")
    keyframe_rate: int = Field(default=1, ge=1, le=10, description="Keyframes per second")
    max_duration: float = Field(default=60.0, ge=1.0, le=600.0, description="Max processing duration")
    generate_thumbnail: bool = Field(default=True, description="Generate thumbnail")
    
class ProcessingOptions(BaseModel):
    """Model for media processing options."""
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    time_filter_hours: Optional[int] = Field(None, ge=1, le=8760)  # Max 1 year
    kind_filter: Optional[str] = Field(None, description="Filter by media kind")
    max_matches: int = Field(default=10, ge=1, le=100)
    
    @validator('similarity_threshold')
    def validate_similarity_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Similarity threshold must be between 0.0 and 1.0')
        return v 