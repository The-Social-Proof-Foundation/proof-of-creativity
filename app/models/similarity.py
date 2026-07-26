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
    similarity_score_percent: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="Same signal as Move/indexer: integer percent 0–100 (rounded from similarity_score)",
    )
    match_type: MatchType = Field(..., description="Type of match")
    confidence_level: ConfidenceLevel = Field(..., description="Confidence level")
    match_details: Optional[Dict[str, Any]] = Field(default=None, description="Additional match details")
    
    class Config:
        use_enum_values = True

class TrackMatchSummary(BaseModel):
    """Per-modality similarity summary used for PoC attestations."""

    similarity_score: float = Field(..., ge=0.0, le=1.0)
    similarity_score_u64: int = Field(..., ge=0, le=100, description="Integer percent mapped for Move thresholds")
    best_match_media_id: Optional[str] = None
    derivative_by_chain_threshold: bool = Field(
        default=False,
        description="True iff score meets on-chain modality threshold AND creator attributable",
    )


class VideoTrackAttestation(BaseModel):
    """Off-chain attestations for VIDEO: visual vs embedded-audio modalities."""

    video_visual: TrackMatchSummary
    embedded_audio: TrackMatchSummary
    embedded_audio_only_derivative_sent: bool = Field(
        default=False,
        description="Matches embedded_audio_only_derivative passed on-chain when applicable.",
    )


class PocChainSummary(BaseModel):
    """Oracle interpretation aligned with Move PoC thresholds (integer percent 0–100)."""

    media_type_code: int = Field(..., description="Move media type: 1 image, 2 video, 3 audio")
    highest_similarity_score_u64: int = Field(..., ge=0, le=100)
    effective_threshold_u64: int = Field(
        ...,
        ge=0,
        le=100,
        description="Threshold used for derivative detection (video+embedded_audio uses audio_threshold)",
    )
    would_apply_derivative_redirect: bool = Field(
        ...,
        description="True iff score >= effective threshold and original_creator is set (non-explicit path)",
    )
    embedded_audio_only_derivative: bool = False
    original_creator: Optional[str] = None
    apply_explicit_outcome: bool = False
    explicit_poc_outcome: int = 0


class UploadResponse(BaseModel):
    """Response model for media upload and processing."""
    media_id: str = Field(..., description="Unique identifier for uploaded media")
    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME type of the uploaded file")
    file_size: int = Field(..., description="File size in bytes")
    file_hash: str = Field(..., description="SHA-256 hash of file content")
    storage_uri: str = Field(..., description="Storage URI (R2, GCS, or local)")
    streaming_uri: Optional[str] = Field(None, description="Cloudflare Stream URI (for videos with upload_to_stream=true)")
    matches: List[MediaMatch] = Field(default=[], description="Similar media found")
    processing_status: str = Field(..., description="Processing status")
    message: str = Field(..., description="Human-readable message")
    tx_hash: Optional[str] = Field(None, description="Transaction hash after PoC submission (when applicable)")
    video_attestation: Optional[VideoTrackAttestation] = Field(
        default=None,
        description="Separate visual vs embedded-audio signals (videos only)",
    )
    poc_chain_summary: Optional[PocChainSummary] = Field(
        default=None,
        description="Chain-aligned score/threshold/creator interpretation for this submission",
    )
    poc_submission_error: Optional[str] = Field(
        default=None,
        description="When post_id PoC RPC submission fails but oracle processing completed",
    )


class StreamingUploadResponse(BaseModel):
    """Response model for streaming upload initiation."""
    upload_id: str = Field(..., description="Unique upload identifier (media_id)")
    predicted_url: str = Field(..., description="Predicted CDN URL (available immediately)")
    websocket_url: str = Field(..., description="WebSocket URL for real-time progress updates")
    message: str = Field(..., description="Status message")
    post_id: Optional[str] = Field(None, description="MySocial post ID if provided")


class PresignUploadRequest(BaseModel):
    """Request body for direct-to-R2 upload reservation (chain-first publish)."""
    filename: str = Field(..., description="Original filename (extension used for object key)")
    content_type: str = Field(..., description="MIME type, e.g. video/mp4")
    content_length: Optional[int] = Field(
        None,
        description="Optional declared file size in bytes (validated against MAX_FILE_SIZE when set)",
    )


class PresignUploadResponse(BaseModel):
    """Presigned PUT slot for client→R2 upload; public_url is safe to put on-chain."""
    media_id: str = Field(..., description="Server-assigned media id (object basename)")
    key: str = Field(..., description="R2 object key, e.g. video/YYYY/MM/{media_id}.mp4")
    public_url: str = Field(..., description="HTTPS CDN URL to use in create_post media_urls")
    upload_url: str = Field(..., description="Presigned PUT URL for the video bytes")
    expires_in: int = Field(..., description="Presigned URL lifetime in seconds")
    content_type: str = Field(..., description="Content-Type the client must send on PUT")


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