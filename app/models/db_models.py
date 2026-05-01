"""
SQLAlchemy database models - the single source of truth for database schema.
Similar to Diesel schema in Rust - migrations are auto-generated from these models.
"""
from sqlalchemy import Column, String, Integer, BigInteger, Float, DateTime, Text, LargeBinary, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector  # pgvector extension
import uuid

Base = declarative_base()

class MediaFile(Base):
    """Media files table - tracks all uploaded media"""
    __tablename__ = 'media_files'
    
    media_id = Column(String(100), primary_key=True, default=lambda: str(uuid.uuid4()))  # Increased for video frames
    filename = Column(String(500), nullable=False)
    original_filename = Column(String(500))
    content_type = Column(String(100))
    file_size = Column(BigInteger)
    storage_uri = Column(Text)
    streaming_uri = Column(Text)  # Cloudflare Stream URI for video streaming (optional, opt-in)
    file_hash = Column(String(128))
    creator_address = Column(String(128))  # MySocial wallet mapped for PoC original_creator lookups
    status = Column(String(20), default='processing')

    processing_results = Column(JSONB, default={})
    matches_found = Column(Integer, default=0)  # Number of similarity matches
    processing_time_ms = Column(Float, default=0.0)  # Processing time in milliseconds
    media_type = Column(String(20))  # 'image', 'audio', 'video'
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_media_files_hash', 'file_hash'),
        Index('idx_media_files_status', 'status'),
        Index('idx_media_files_created_at', 'created_at'),
    )

class MediaEmbedding(Base):
    """CLIP embeddings for visual similarity search"""
    __tablename__ = 'media_embeddings'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    media_id = Column(String(100), nullable=False)  # Increased for video frames: uuid_frame_idx
    kind = Column(String(50), nullable=False)  # 'image', 'video_frame', 'audio'
    embedding = Column(Vector(512), nullable=False)  # pgvector type
    meta = Column('metadata', JSONB, default={})  # renamed 'metadata' -> 'meta' (reserved word)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_media_embeddings_media_id', 'media_id'),
        Index('idx_media_embeddings_kind', 'kind'),
        Index('idx_media_embeddings_uploaded_at', 'uploaded_at'),
        # Vector index created separately via raw SQL (pgvector specific)
    )

class AudioFingerprint(Base):
    """Audio fingerprints for audio similarity detection"""
    __tablename__ = 'audio_fingerprints'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    fp_hash = Column(String(128), nullable=False)
    media_id = Column(String(100), nullable=False)  # Increased for video frames
    offset_seconds = Column(Float, default=0.0)
    fingerprint_data = Column(LargeBinary)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_audio_fingerprints_hash', 'fp_hash'),
        Index('idx_audio_fingerprints_media_id', 'media_id', 'offset_seconds'),
        Index('idx_audio_fingerprints_created_at', 'created_at'),
    )

class ImageHash(Base):
    """Perceptual image hashes for duplicate detection"""
    __tablename__ = 'image_hashes'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    media_id = Column(String(100), nullable=False, unique=True)  # Increased for consistency
    dhash = Column(String(64))
    phash = Column(String(64))
    ahash = Column(String(64))
    dhash_16 = Column(String(256))
    phash_16 = Column(String(256))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_image_hashes_media_id', 'media_id'),
        Index('idx_image_hashes_dhash', 'dhash'),
        Index('idx_image_hashes_phash', 'phash'),
    )

class SimilarityMatch(Base):
    """Similarity matches between media files"""
    __tablename__ = 'similarity_matches'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_media_id = Column(String(100), nullable=False)  # Increased for video frames
    match_media_id = Column(String(100), nullable=False)  # Increased for video frames
    match_type = Column(String(50), nullable=False)  # 'embedding', 'fingerprint', 'perceptual_hash'
    similarity_score = Column(Float, nullable=False)
    confidence_level = Column(String(20), default='medium')
    match_details = Column(JSONB, default={})
    match_category = Column(String(50))  # Additional categorization
    embedding_type = Column(String(50))  # Type of embedding used
    fingerprint_hash = Column(String(128))  # For fingerprint matches
    offset_seconds = Column(Float)  # Time offset for audio matches
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    __table_args__ = (
        Index('idx_similarity_matches_query_media', 'query_media_id', 'created_at'),
        Index('idx_similarity_matches_score', 'similarity_score'),
        Index('idx_similarity_matches_type_score', 'match_type', 'similarity_score'),
    )

class AttributionRecord(Base):
    """Blockchain attribution records"""
    __tablename__ = 'attribution_records'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    media_id = Column(String(100), nullable=False)  # Increased for video frames
    tx_hash = Column(String(128))
    wallet_address = Column(String(128))
    attribution_type = Column(String(50))  # 'original', 'derivative', 'remix'
    proof_data = Column(JSONB, default={})
    file_hash = Column(String(128))  # SHA-256 hash of file
    media_type = Column(String(20))  # 'image', 'audio', 'video'
    matches_found = Column(Integer, default=0)  # Number of matches
    high_confidence_matches = Column(Integer, default=0)  # High confidence count
    max_similarity_score = Column(Float, default=0.0)  # Highest similarity found
    processing_time_ms = Column(Float, default=0.0)  # Processing time
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    __table_args__ = (
        Index('idx_attribution_records_media', 'media_id', 'created_at'),
        Index('idx_attribution_records_blockchain', 'tx_hash'),
    )

