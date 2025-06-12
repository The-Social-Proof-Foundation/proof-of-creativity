-- Enhanced Database Schema for Proof of Creativity Media Attribution System
-- Optimized for Timescale Vector AI

-- Enable required extensions for Timescale + Vector AI
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Table for storing media embeddings (images, video frames) with time-series optimization
CREATE TABLE IF NOT EXISTS media_embeddings (
    id UUID DEFAULT gen_random_uuid(),
    media_id VARCHAR(36) NOT NULL, -- Changed to VARCHAR for consistency
    kind VARCHAR(50) NOT NULL, -- 'image', 'video_frame', 'audio'
    embedding VECTOR(512) NOT NULL, -- CLIP embeddings are 512-dimensional
    metadata JSONB DEFAULT '{}',
    uploaded_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, uploaded_at) -- Composite primary key for hypertable
);

-- Convert to hypertable for time-series + vector performance
SELECT create_hypertable('media_embeddings', 'uploaded_at', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Table for storing audio fingerprints with time-series support
CREATE TABLE IF NOT EXISTS audio_fingerprints (
    id UUID DEFAULT gen_random_uuid(),
    fp_hash VARCHAR(128) NOT NULL, -- SHA-256 hash of fingerprint
    media_id VARCHAR(36) NOT NULL,
    offset_seconds FLOAT DEFAULT 0.0, -- Time offset in the audio file
    fingerprint_data BYTEA, -- Raw fingerprint data for detailed matching
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, created_at) -- Composite primary key for hypertable
);

-- Convert fingerprints to hypertable
SELECT create_hypertable('audio_fingerprints', 'created_at', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Table for storing media metadata with time-series capabilities
CREATE TABLE IF NOT EXISTS media_files (
    media_id VARCHAR(36),
    filename VARCHAR(500) NOT NULL,
    original_filename VARCHAR(500),
    content_type VARCHAR(100),
    file_size BIGINT,
    storage_uri TEXT, -- GCS or Walrus URI
    file_hash VARCHAR(128), -- SHA-256 hash of file content
    upload_user_id VARCHAR(36), -- For future user authentication
    upload_ip INET,
    status VARCHAR(20) DEFAULT 'processing', -- 'processing', 'completed', 'failed'
    processing_results JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (media_id, created_at) -- Composite primary key for hypertable
);

-- Convert media files to hypertable for time-series analytics
SELECT create_hypertable('media_files', 'created_at', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Table for storing similarity matches with time-series analytics
CREATE TABLE IF NOT EXISTS similarity_matches (
    id UUID DEFAULT gen_random_uuid(),
    query_media_id VARCHAR(36) NOT NULL,
    match_media_id VARCHAR(36) NOT NULL,
    match_type VARCHAR(50) NOT NULL, -- 'embedding', 'fingerprint', 'exact'
    similarity_score FLOAT NOT NULL, -- 0.0 to 1.0
    confidence_level VARCHAR(20) DEFAULT 'medium', -- 'low', 'medium', 'high'
    match_details JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, created_at) -- Composite primary key for hypertable
);

-- Convert similarity matches to hypertable for analytics
SELECT create_hypertable('similarity_matches', 'created_at', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Table for blockchain attribution records with time-series support
CREATE TABLE IF NOT EXISTS attribution_records (
    id UUID DEFAULT gen_random_uuid(),
    media_id VARCHAR(36) NOT NULL,
    blockchain_tx_hash VARCHAR(128),
    blockchain_address VARCHAR(128),
    attribution_type VARCHAR(50), -- 'original', 'derivative', 'remix'
    proof_data JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, created_at) -- Composite primary key for hypertable
);

-- Convert attribution records to hypertable
SELECT create_hypertable('attribution_records', 'created_at', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- ENHANCED INDEXES FOR TIMESCALE VECTOR AI PERFORMANCE

-- Vector similarity search with HNSW index (optimal for Timescale Vector AI)
CREATE INDEX IF NOT EXISTS idx_media_embeddings_vector_hnsw 
ON media_embeddings USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);

-- Fallback IVFFlat index for broader compatibility
CREATE INDEX IF NOT EXISTS idx_media_embeddings_vector_ivf 
ON media_embeddings USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- Time-based and composite indexes for efficient filtering
CREATE INDEX IF NOT EXISTS idx_media_embeddings_uploaded_at 
ON media_embeddings (uploaded_at DESC);

CREATE INDEX IF NOT EXISTS idx_media_embeddings_kind 
ON media_embeddings (kind);

CREATE INDEX IF NOT EXISTS idx_media_embeddings_kind_uploaded 
ON media_embeddings (kind, uploaded_at DESC);

-- Time-series + vector composite index for advanced queries
CREATE INDEX IF NOT EXISTS idx_media_embeddings_kind_time_vector
ON media_embeddings (kind, uploaded_at DESC, embedding);

-- Audio fingerprint optimization
CREATE INDEX IF NOT EXISTS idx_audio_fingerprints_hash 
ON audio_fingerprints (fp_hash);

CREATE INDEX IF NOT EXISTS idx_audio_fingerprints_media_offset 
ON audio_fingerprints (media_id, offset_seconds);

-- Hash-based deduplication index
CREATE INDEX IF NOT EXISTS idx_audio_fingerprints_hash_created 
ON audio_fingerprints (fp_hash, created_at DESC);

-- Media file optimization
CREATE INDEX IF NOT EXISTS idx_media_files_hash 
ON media_files (file_hash);

CREATE INDEX IF NOT EXISTS idx_media_files_created_at 
ON media_files (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_media_files_status 
ON media_files (status);

-- Similarity search optimization
CREATE INDEX IF NOT EXISTS idx_similarity_matches_query_media 
ON similarity_matches (query_media_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_similarity_matches_score 
ON similarity_matches (similarity_score DESC);

CREATE INDEX IF NOT EXISTS idx_similarity_matches_type_score 
ON similarity_matches (match_type, similarity_score DESC);

-- Attribution record optimization
CREATE INDEX IF NOT EXISTS idx_attribution_records_media 
ON attribution_records (media_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_attribution_records_blockchain 
ON attribution_records (blockchain_tx_hash);

-- TIMESCALE-SPECIFIC OPTIMIZATIONS

-- Continuous aggregates for analytics (optional, can be added later)
-- CREATE MATERIALIZED VIEW hourly_upload_stats
-- WITH (timescaledb.continuous) AS
-- SELECT 
--     time_bucket('1 hour', created_at) AS hour,
--     COUNT(*) as uploads,
--     COUNT(DISTINCT content_type) as content_types
-- FROM media_files
-- GROUP BY hour;

-- Data retention policies (customize based on needs)
-- SELECT add_retention_policy('media_files', INTERVAL '1 year');
-- SELECT add_retention_policy('similarity_matches', INTERVAL '6 months');

-- Compression policies for older data
-- SELECT add_compression_policy('media_embeddings', INTERVAL '30 days');
-- SELECT add_compression_policy('audio_fingerprints', INTERVAL '30 days');

-- Update triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_media_embeddings_updated_at 
    BEFORE UPDATE ON media_embeddings 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_audio_fingerprints_updated_at 
    BEFORE UPDATE ON audio_fingerprints 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_media_files_updated_at 
    BEFORE UPDATE ON media_files 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Enhanced comments for documentation
COMMENT ON TABLE media_embeddings IS 'Timescale hypertable storing CLIP embeddings with time-series optimization';
COMMENT ON TABLE audio_fingerprints IS 'Timescale hypertable for audio fingerprints with time-series analytics';
COMMENT ON TABLE media_files IS 'Timescale hypertable for media metadata with upload tracking';
COMMENT ON TABLE similarity_matches IS 'Timescale hypertable recording similarity matches with temporal analytics';
COMMENT ON TABLE attribution_records IS 'Timescale hypertable for blockchain attribution with time-series support';

COMMENT ON COLUMN media_embeddings.embedding IS 'CLIP ViT-B/32 512-dimensional embedding vector with HNSW index';
COMMENT ON COLUMN audio_fingerprints.fp_hash IS 'SHA-256 hash of audio fingerprint for fast lookup';
COMMENT ON COLUMN media_files.file_hash IS 'SHA-256 hash of original file content for deduplication';
COMMENT ON COLUMN similarity_matches.similarity_score IS 'Cosine similarity score between 0.0 and 1.0';

-- Initial data validation and constraints
ALTER TABLE media_embeddings ADD CONSTRAINT check_similarity_score_range 
    CHECK (vector_dims(embedding) = 512);

ALTER TABLE similarity_matches ADD CONSTRAINT check_similarity_score_range 
    CHECK (similarity_score >= 0.0 AND similarity_score <= 1.0);

ALTER TABLE similarity_matches ADD CONSTRAINT check_confidence_level 
    CHECK (confidence_level IN ('low', 'medium', 'high'));

ALTER TABLE media_files ADD CONSTRAINT check_status_values 
    CHECK (status IN ('processing', 'completed', 'failed'));

ALTER TABLE attribution_records ADD CONSTRAINT check_attribution_type 
    CHECK (attribution_type IN ('original', 'derivative', 'remix', 'licensed'));

-- Performance monitoring views
CREATE OR REPLACE VIEW vector_search_performance AS
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan as searches_performed,
    idx_tup_read as tuples_read,
    idx_tup_fetch as tuples_fetched
FROM pg_stat_user_indexes 
WHERE indexname LIKE '%vector%'
ORDER BY idx_scan DESC;

CREATE OR REPLACE VIEW daily_upload_summary AS
SELECT 
    DATE(created_at) as upload_date,
    COUNT(*) as total_uploads,
    COUNT(DISTINCT content_type) as content_types,
    SUM(file_size) as total_bytes,
    AVG(file_size) as avg_file_size
FROM media_files
WHERE created_at >= NOW() - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY upload_date DESC;

-- Grant permissions (adjust for your user)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO your_app_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO your_app_user; 