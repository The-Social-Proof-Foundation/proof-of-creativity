"""
SQLAlchemy database models - the single source of truth for database schema.
Similar to Diesel schema in Rust - migrations are auto-generated from these models.
"""
from sqlalchemy import Column, String, Integer, BigInteger, Float, DateTime, Text, LargeBinary, Index, Boolean
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
    media_id = Column(String(100), nullable=False)
    kind = Column(String(50), nullable=False)
    embedding = Column(Vector(512), nullable=False)
    meta = Column('metadata', JSONB, default={})
    corpus_scope = Column(String(32), default='platform')
    discovery_asset_id = Column(UUID(as_uuid=True))
    embedding_model = Column(String(128))
    embedding_version = Column(String(64))
    embedding_dimension = Column(Integer)
    embedding_created_at = Column(DateTime(timezone=True))
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_media_embeddings_media_id', 'media_id'),
        Index('idx_media_embeddings_kind', 'kind'),
        Index('idx_media_embeddings_uploaded_at', 'uploaded_at'),
        Index('idx_media_embeddings_corpus_scope', 'corpus_scope'),
    )

class AudioFingerprint(Base):
    """Audio fingerprints for audio similarity detection"""
    __tablename__ = 'audio_fingerprints'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    fp_hash = Column(String(128), nullable=False)
    media_id = Column(String(100), nullable=False)  # Increased for video frames
    offset_seconds = Column(Float, default=0.0)
    fingerprint_data = Column(LargeBinary)
    corpus_scope = Column(String(32), default='platform')
    discovery_asset_id = Column(UUID(as_uuid=True))
    embedding_model = Column(String(128))
    embedding_version = Column(String(64))
    embedding_dimension = Column(Integer)
    embedding_created_at = Column(DateTime(timezone=True))
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
    corpus_scope = Column(String(32), default='platform')
    discovery_asset_id = Column(UUID(as_uuid=True))
    embedding_model = Column(String(128))
    embedding_version = Column(String(64))
    embedding_dimension = Column(Integer)
    embedding_created_at = Column(DateTime(timezone=True))
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


# --- Oracle / chain sync tables (network-scoped) ---


class ChainPost(Base):
    __tablename__ = "chain_posts"

    network = Column(String(20), primary_key=True)
    post_id = Column(String(128), primary_key=True)
    creator_address = Column(String(128))
    enable_poc = Column(String(10), default="true")
    media_urls = Column(JSONB, default=list)
    media_types = Column(JSONB, default=list)
    media_asset_ids = Column(JSONB, default=list)
    composition_status = Column(Integer)
    monetization_status = Column(Integer)
    analysis_status = Column(String(32), default="discovered")
    poc_outcome = Column(Integer)
    highest_similarity_score = Column(Integer)
    proof_bundle_uri = Column(Text)
    tx_digest = Column(String(128))
    discovered_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    metadata_json = Column("metadata", JSONB, default=dict)


class GrpcSyncCheckpoint(Base):
    __tablename__ = "grpc_sync_checkpoints"

    network = Column(String(20), primary_key=True)
    # 0x + 64 hex (full Move address); short forms like 0x50c1 also fit
    stream_id = Column(String(66), primary_key=True, default="default")
    checkpoint_sequence = Column(BigInteger, default=0)
    last_transaction_digest = Column(String(128))
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class OracleJob(Base):
    __tablename__ = "oracle_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    network = Column(String(20), nullable=False)
    post_id = Column(String(128), nullable=False)
    job_type = Column(String(32), nullable=False, default="analyze_post")
    media_url = Column(Text)
    media_index = Column(Integer, default=0)
    media_type = Column(Integer)
    status = Column(String(32), default="pending")
    attempts = Column(Integer, default=0)
    last_error = Column(Text)
    payload = Column(JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_oracle_jobs_status_network", "network", "status", "created_at"),
        Index("idx_oracle_jobs_post", "network", "post_id"),
    )


class ChainAttestation(Base):
    __tablename__ = "chain_attestations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    network = Column(String(20), nullable=False)
    post_id = Column(String(128), nullable=False)
    tx_digest = Column(String(128))
    media_type = Column(Integer)
    highest_similarity_score = Column(Integer)
    original_creator = Column(String(128))
    derivative_redirection_target = Column(Integer)
    poc_outcome = Column(Integer)
    reasoning = Column(Text)
    evidence_urls = Column(JSONB, default=list)
    status = Column(String(32), default="submitted")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("idx_chain_attestations_post", "network", "post_id", "created_at"),
    )


class MediaPostLink(Base):
    __tablename__ = "media_post_links"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    network = Column(String(20), nullable=False)
    post_id = Column(String(128), nullable=False)
    media_id = Column(String(100), nullable=False)
    media_url = Column(Text)
    media_index = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("idx_media_post_links_post", "network", "post_id"),
        Index("idx_media_post_links_media", "media_id"),
    )


class UsernameBeneficiary(Base):
    __tablename__ = "username_beneficiaries"

    network = Column(String(20), primary_key=True)
    identity_hash = Column(String(128), primary_key=True)
    username = Column(String(256))
    beneficiary_address = Column(String(128))
    vault_object_id = Column(String(128))
    provision_tx_digest = Column(String(128))
    claimed = Column(String(10), default="false")
    claim_tx_digest = Column(String(128))
    metadata_json = Column("metadata", JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class ChainConfigCache(Base):
    __tablename__ = "chain_config_cache"

    network = Column(String(20), primary_key=True)
    config_json = Column(JSONB, default=dict)
    fetched_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class ExternalIdentity(Base):
    __tablename__ = "external_identities"

    network = Column(String(20), primary_key=True)
    identity_hash = Column(String(128), primary_key=True)
    platform = Column(String(64))
    external_id = Column(String(256))
    display_name = Column(String(256))
    metadata_json = Column("metadata", JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class ProvenanceHit(Base):
    __tablename__ = "provenance_hits"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    network = Column(String(20), nullable=False)
    post_id = Column(String(128), nullable=False)
    query_media_id = Column(String(100))
    discovery_asset_id = Column(UUID(as_uuid=True))
    creator_candidate_id = Column(UUID(as_uuid=True))
    similarity_score = Column(Float, default=0.0)
    match_type = Column(String(50))
    work_confidence = Column(Float, default=0.0)
    creator_confidence = Column(Float, default=0.0)
    decision = Column(String(32), default="pending")
    vault_provisioned = Column(Boolean, default=False)
    vault_identity_hash = Column(String(128))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("idx_provenance_hits_post", "network", "post_id", "created_at"),
    )


class DiscoverySource(Base):
    __tablename__ = "discovery_sources"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    adapter_type = Column(String(64), nullable=False)
    domain = Column(String(32), default="creative")
    source_url = Column(Text)
    config = Column(JSONB, default=dict)
    trust_score = Column(Float, default=0.5)
    enabled = Column(Boolean, default=True)
    terms_notes = Column(Text)
    last_polled_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class CreatorCandidate(Base):
    __tablename__ = "creator_candidates"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    primary_x_handle = Column(String(256), unique=True)
    identity_hash = Column(String(128))
    display_name = Column(String(256))
    aliases = Column(JSONB, default=list)
    platform_handles = Column(JSONB, default=dict)
    source_urls = Column(JSONB, default=list)
    creator_confidence = Column(Float, default=0.0)
    work_count = Column(Integer, default=0)
    blockchain_hit_count = Column(Integer, default=0)
    similarity_hit_count = Column(Integer, default=0)
    lifecycle_state = Column(String(32), default="unresolved")
    merge_target_id = Column(UUID(as_uuid=True))
    metadata_json = Column("metadata", JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class DiscoveryAsset(Base):
    __tablename__ = "discovery_assets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id = Column(UUID(as_uuid=True))
    external_source_url = Column(Text, nullable=False, unique=True)
    canonical_metadata = Column(JSONB, default=dict)
    media_type = Column(String(32), nullable=False)
    content_kind = Column(String(16), default="media")
    content_hash = Column(String(128))
    metadata_hash = Column(String(128))
    lifecycle_state = Column(String(32), default="discovered")
    source_trust_score = Column(Float, default=0.5)
    work_confidence = Column(Float, default=0.0)
    creator_confidence = Column(Float, default=0.0)
    creator_candidate_id = Column(UUID(as_uuid=True))
    active_embedding_version = Column(String(64))
    related_on_chain_post = Column(String(128))
    priority_score = Column(BigInteger, default=0)
    discovered_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    exclusion_reason = Column(Text)

    __table_args__ = (
        Index("idx_discovery_assets_lifecycle", "lifecycle_state", "priority_score"),
        Index("idx_discovery_assets_creator", "creator_candidate_id"),
    )


class CompositionAnalysisRecord(Base):
    __tablename__ = "composition_analysis_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    network = Column(String(20), nullable=False)
    post_id = Column(String(128), nullable=False)
    composition_status = Column(Integer)
    monetization_status = Column(Integer)
    analysis_json = Column(JSONB, default=dict)
    manifest_json = Column(JSONB)
    reasoning = Column(Text)
    evidence_urls = Column(JSONB, default=list)
    contains_derivatives = Column(Boolean, default=False)
    contains_unresolved_assets = Column(Boolean, default=False)
    tx_digest = Column(String(128))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("idx_composition_analysis_post", "network", "post_id", "created_at"),
    )


class ChainMediaAsset(Base):
    __tablename__ = "chain_media_assets"

    network = Column(String(20), primary_key=True)
    asset_id = Column(String(128), primary_key=True)
    content_commitment = Column(String(256))
    fingerprint_commitment = Column(String(256))
    media_type = Column(Integer)
    asset_kind = Column(Integer, default=0)
    originality_status = Column(Integer)
    lineage_parent_id = Column(String(128))
    pending_id = Column(String(128))
    policy_version = Column(Integer, default=0)
    creators = Column(JSONB, default=list)
    rights_controllers = Column(JSONB, default=list)
    beneficiaries = Column(JSONB, default=list)
    beneficiary_splits = Column(JSONB, default=list)
    rights_json = Column(JSONB, default=dict)
    rights_version = Column(Integer, default=1)
    economics_version = Column(Integer, default=1)
    linked_existing = Column(Boolean, default=False)
    resolve_tx_digest = Column(String(128))
    request_id = Column(String(128))
    metadata_json = Column("metadata", JSONB, default=dict)
    registered_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_chain_media_assets_fingerprint", "network", "fingerprint_commitment"),
    )


class FingerprintObservation(Base):
    __tablename__ = "fingerprint_observations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    network = Column(String(20), nullable=False)
    fingerprint_commitment = Column(String(256), nullable=False)
    content_commitment = Column(String(256))
    media_asset_id = Column(String(128))
    request_id = Column(String(128))
    media_type = Column(Integer)
    submitter = Column(String(128))
    observed_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("idx_fingerprint_observations_fp", "network", "fingerprint_commitment"),
        Index("idx_fingerprint_observations_asset", "network", "media_asset_id"),
    )


class MediaAssetUsage(Base):
    __tablename__ = "media_asset_usages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    network = Column(String(20), nullable=False)
    asset_id = Column(String(128), nullable=False)
    container_id = Column(String(128), nullable=False)
    container_type = Column(Integer, default=1)
    usage_class = Column(Integer, default=1)
    position = Column(Integer, default=0)
    tx_digest = Column(String(128))
    observed_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("idx_media_asset_usages_asset", "network", "asset_id", "observed_at"),
        Index("idx_media_asset_usages_container", "network", "container_id"),
    )


class PendingDerivativeAsset(Base):
    __tablename__ = "pending_derivative_assets"

    network = Column(String(20), primary_key=True)
    pending_id = Column(String(128), primary_key=True)
    request_id = Column(String(128))
    content_commitment = Column(String(256))
    fingerprint_commitment = Column(String(256))
    media_type = Column(Integer)
    asset_kind = Column(Integer, default=0)
    creator = Column(String(128))
    status = Column(String(32), default="pending")
    finalize_tx_digest = Column(String(128))
    child_asset_id = Column(String(128))
    metadata_json = Column("metadata", JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_pending_derivative_request", "network", "request_id"),
    )


class DerivativeEdge(Base):
    __tablename__ = "derivative_edges"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    network = Column(String(20), nullable=False)
    parent_asset_id = Column(String(128), nullable=False)
    child_asset_id = Column(String(128), nullable=False)
    relationship_type = Column(Integer, default=1)
    license_instance_id = Column(String(128))
    template_version_id = Column(String(128))
    parent_share_bps = Column(Integer)
    tx_digest = Column(String(128))
    observed_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("idx_derivative_edges_child", "network", "child_asset_id"),
        Index("idx_derivative_edges_parent", "network", "parent_asset_id"),
    )


class DetectedAssetRelationship(Base):
    __tablename__ = "detected_asset_relationships"

    network = Column(String(20), primary_key=True)
    proposal_id = Column(String(128), primary_key=True)
    accused_pending_id = Column(String(128), nullable=False)
    accused_asset_id = Column(String(128))
    original_asset_id = Column(String(128), nullable=False)
    similarity_bps = Column(Integer, nullable=False)
    status = Column(Integer, default=0)
    evidence_commitment = Column(String(256))
    tx_digest = Column(String(128))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_detected_relationships_pending", "network", "accused_pending_id"),
    )


class PostEnforcementSnapshot(Base):
    __tablename__ = "post_enforcement_snapshots"

    network = Column(String(20), primary_key=True)
    post_id = Column(String(128), primary_key=True)
    bindings_json = Column(JSONB, default=list)
    usage_decisions_json = Column(JSONB, default=list)
    usage_denials_json = Column(JSONB, default=list)
    playback_policy_json = Column(JSONB, default=dict)
    tx_digest = Column(String(128))
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class DiscoveryJob(Base):
    __tablename__ = "discovery_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_type = Column(String(32), nullable=False)
    discovery_asset_id = Column(UUID(as_uuid=True))
    priority_score = Column(BigInteger, default=0)
    status = Column(String(32), default="pending")
    attempts = Column(Integer, default=0)
    max_attempts = Column(Integer, default=5)
    run_after = Column(DateTime(timezone=True), server_default=func.now())
    last_error = Column(Text)
    payload = Column(JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_discovery_jobs_claim", "status", "run_after", "priority_score", "created_at"),
    )

