-- Clean Timescale-optimized Database Schema for Proof of Creativity
-- This script drops existing tables and creates Timescale-compatible schema

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Drop existing tables if they exist (in reverse dependency order)
DROP TABLE IF EXISTS attribution_records CASCADE;
DROP TABLE IF EXISTS similarity_matches CASCADE;
DROP TABLE IF EXISTS media_embeddings CASCADE;
DROP TABLE IF EXISTS audio_fingerprints CASCADE;
DROP TABLE IF EXISTS media_files CASCADE;

-- Table 1: Media Files (main metadata table)
CREATE TABLE media_files (
    media_id VARCHAR(36) NOT NULL,
    filename VARCHAR(500) NOT NULL,
    original_filename VARCHAR(500),
    content_type VARCHAR(100),
    file_size BIGINT,
    storage_uri TEXT,
    file_hash VARCHAR(128),
    upload_user_id VARCHAR(36),
    upload_ip INET,
    status VARCHAR(20) DEFAULT 'processing',
    processing_results JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create hypertable for media_files
SELECT create_hypertable('media_files', 'created_at', 
    chunk_time_interval => INTERVAL '1 day'
);

-- Add unique constraint on media_id (Timescale-compatible)
CREATE UNIQUE INDEX idx_media_files_media_id_created ON media_files(media_id, created_at);

-- Table 2: Media Embeddings (vector storage)
CREATE TABLE media_embeddings (
    id UUID DEFAULT gen_random_uuid(),
    media_id VARCHAR(36) NOT NULL,
    kind VARCHAR(50) NOT NULL,
    embedding VECTOR(512) NOT NULL,
    metadata JSONB DEFAULT '{}',
    uploaded_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create hypertable for embeddings
SELECT create_hypertable('media_embeddings', 'uploaded_at', 
    chunk_time_interval => INTERVAL '1 day'
);

-- Table 3: Audio Fingerprints
CREATE TABLE audio_fingerprints (
    id UUID DEFAULT gen_random_uuid(),
    fp_hash VARCHAR(128) NOT NULL,
    media_id VARCHAR(36) NOT NULL,
    offset_seconds FLOAT DEFAULT 0.0,
    fingerprint_data BYTEA,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create hypertable for fingerprints
SELECT create_hypertable('audio_fingerprints', 'created_at', 
    chunk_time_interval => INTERVAL '1 day'
);

-- Table 4: Image Perceptual Hashes (Regular table for fast exact lookups)
CREATE TABLE image_hashes (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    media_id VARCHAR(36) NOT NULL UNIQUE,
    dhash VARCHAR(32),       -- 8x8 difference hash (64 bits = 16 hex chars)
    phash VARCHAR(32),       -- 8x8 perceptual hash  
    ahash VARCHAR(32),       -- 8x8 average hash
    dhash_16 VARCHAR(128),   -- 16x16 difference hash (256 bits = 64 hex chars)
    phash_16 VARCHAR(128),   -- 16x16 perceptual hash
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Add indexes for image hash searches
CREATE INDEX idx_image_hashes_dhash ON image_hashes (dhash);
CREATE INDEX idx_image_hashes_phash ON image_hashes (phash);
CREATE INDEX idx_image_hashes_ahash ON image_hashes (ahash);
CREATE INDEX idx_image_hashes_dhash_16 ON image_hashes (dhash_16);
CREATE INDEX idx_image_hashes_phash_16 ON image_hashes (phash_16);
CREATE INDEX idx_image_hashes_media_id ON image_hashes (media_id);

-- Table 5: Similarity Matches
CREATE TABLE similarity_matches (
    id UUID DEFAULT gen_random_uuid(),
    query_media_id VARCHAR(36) NOT NULL,
    match_media_id VARCHAR(36) NOT NULL,
    match_type VARCHAR(50) NOT NULL,
    similarity_score FLOAT NOT NULL,
    confidence_level VARCHAR(20) DEFAULT 'medium',
    match_details JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create hypertable for similarity matches
SELECT create_hypertable('similarity_matches', 'created_at', 
    chunk_time_interval => INTERVAL '1 day'
);

-- Table 6: Attribution Records
CREATE TABLE attribution_records (
    id UUID DEFAULT gen_random_uuid(),
    media_id VARCHAR(36) NOT NULL,
    blockchain_tx_hash VARCHAR(128),
    blockchain_address VARCHAR(128),
    attribution_type VARCHAR(50),
    proof_data JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create hypertable for attribution records
SELECT create_hypertable('attribution_records', 'created_at', 
    chunk_time_interval => INTERVAL '1 day'
);

-- INDEXES FOR OPTIMAL PERFORMANCE

-- Vector similarity search with HNSW (best for Timescale Vector AI)
CREATE INDEX idx_media_embeddings_vector_hnsw 
ON media_embeddings USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);

-- Time-based indexes
CREATE INDEX idx_media_embeddings_uploaded_at ON media_embeddings (uploaded_at DESC);
CREATE INDEX idx_media_embeddings_kind ON media_embeddings (kind);
CREATE INDEX idx_media_embeddings_kind_uploaded ON media_embeddings (kind, uploaded_at DESC);

-- Audio fingerprint indexes
CREATE INDEX idx_audio_fingerprints_hash ON audio_fingerprints (fp_hash);
CREATE INDEX idx_audio_fingerprints_media_id ON audio_fingerprints (media_id);
CREATE INDEX idx_audio_fingerprints_hash_created ON audio_fingerprints (fp_hash, created_at DESC);

-- Media file indexes
CREATE INDEX idx_media_files_hash ON media_files (file_hash);
CREATE INDEX idx_media_files_created_at ON media_files (created_at DESC);
CREATE INDEX idx_media_files_status ON media_files (status);

-- Similarity match indexes
CREATE INDEX idx_similarity_matches_query_media ON similarity_matches (query_media_id, created_at DESC);
CREATE INDEX idx_similarity_matches_score ON similarity_matches (similarity_score DESC);
CREATE INDEX idx_similarity_matches_type ON similarity_matches (match_type);

-- Attribution indexes
CREATE INDEX idx_attribution_records_media ON attribution_records (media_id, created_at DESC);
CREATE INDEX idx_attribution_records_blockchain ON attribution_records (blockchain_tx_hash);

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

CREATE TRIGGER update_image_hashes_updated_at 
    BEFORE UPDATE ON image_hashes 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Constraints and validations
ALTER TABLE similarity_matches ADD CONSTRAINT check_similarity_score_range 
    CHECK (similarity_score >= 0.0 AND similarity_score <= 1.0);

ALTER TABLE similarity_matches ADD CONSTRAINT check_confidence_level 
    CHECK (confidence_level IN ('low', 'medium', 'high'));

ALTER TABLE media_files ADD CONSTRAINT check_status_values 
    CHECK (status IN ('processing', 'completed', 'failed'));

ALTER TABLE attribution_records ADD CONSTRAINT check_attribution_type 
    CHECK (attribution_type IN ('original', 'derivative', 'remix', 'licensed'));

-- Comments for documentation
COMMENT ON TABLE media_embeddings IS 'Timescale hypertable storing CLIP embeddings with vector search';
COMMENT ON TABLE audio_fingerprints IS 'Timescale hypertable for audio fingerprints';
COMMENT ON TABLE media_files IS 'Timescale hypertable for media metadata';
COMMENT ON TABLE similarity_matches IS 'Timescale hypertable for similarity match records';
COMMENT ON TABLE attribution_records IS 'Timescale hypertable for blockchain attribution';

-- Performance monitoring view
CREATE OR REPLACE VIEW vector_search_performance AS
SELECT 
    schemaname,
    relname as tablename,
    indexrelname as indexname,
    idx_scan as searches_performed,
    idx_tup_read as tuples_read,
    idx_tup_fetch as tuples_fetched
FROM pg_stat_user_indexes 
WHERE indexrelname LIKE '%vector%'
ORDER BY idx_scan DESC; 