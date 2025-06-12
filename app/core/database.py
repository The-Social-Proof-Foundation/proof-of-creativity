import os
import psycopg2
import structlog
from typing import List, Any, Dict, Optional, Tuple
from psycopg2 import extras
from psycopg2.pool import SimpleConnectionPool
from contextlib import contextmanager
import uuid
from datetime import datetime, timedelta

logger = structlog.get_logger()

# Timescale Vector AI configuration
TIMESCALE_DB_DSN = os.getenv("TIMESCALE_DB_DSN", os.getenv("DB_DSN", "postgresql://user:password@localhost:5432/proof_of_creativity"))
TIMESCALE_SERVICE_URL = os.getenv("TIMESCALE_SERVICE_URL", "")
TIMESCALE_API_KEY = os.getenv("TIMESCALE_API_KEY", "")

# Connection pool configuration
MIN_CONNECTIONS = 1
MAX_CONNECTIONS = 20

# Global connection pool
_connection_pool = None

def initialize_connection_pool():
    """Initialize the Timescale database connection pool."""
    global _connection_pool
    if _connection_pool is None:
        try:
            _connection_pool = SimpleConnectionPool(
                MIN_CONNECTIONS, 
                MAX_CONNECTIONS, 
                TIMESCALE_DB_DSN
            )
            logger.info("Timescale connection pool initialized", 
                       min_connections=MIN_CONNECTIONS, 
                       max_connections=MAX_CONNECTIONS)
            
            # Test vector extension
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
                    result = cur.fetchone()
                    if result:
                        logger.info("Vector extension confirmed available")
                    else:
                        logger.warning("Vector extension not found - ensure pgvector is installed")
                        
        except Exception as e:
            logger.error("Failed to initialize Timescale connection pool", error=str(e))
            raise

@contextmanager
def get_db_connection():
    """Context manager for Timescale database connections with automatic cleanup."""
    if _connection_pool is None:
        initialize_connection_pool()
    
    conn = None
    try:
        conn = _connection_pool.getconn()
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error("Database operation failed", error=str(e))
        raise
    finally:
        if conn:
            _connection_pool.putconn(conn)

def get_conn():
    """Legacy function for backward compatibility."""
    return psycopg2.connect(TIMESCALE_DB_DSN)

# Embedding management with Timescale Vector AI
def insert_embedding(media_id: str, kind: str, embedding: List[float], metadata: Dict[str, Any]):
    """Insert a new embedding into Timescale with vector indexing."""
    sql = """
    INSERT INTO media_embeddings (media_id, kind, embedding, metadata)
    VALUES (%s, %s, %s, %s)
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (media_id, kind, embedding, extras.Json(metadata)))
                conn.commit()
        
        logger.debug("Embedding inserted successfully", 
                    media_id=media_id, kind=kind, embedding_dim=len(embedding))
                    
    except Exception as e:
        logger.error("Failed to insert embedding", 
                    media_id=media_id, kind=kind, error=str(e))
        raise

def search_embedding_with_timescale_ai(
    vector: List[float], 
    top_k: int = 5, 
    kind_filter: Optional[str] = None,
    time_filter_hours: Optional[int] = None,
    similarity_threshold: float = 0.7
) -> List[Tuple]:
    """
    Advanced vector similarity search using Timescale Vector AI capabilities.
    
    Args:
        vector: Query embedding vector
        top_k: Number of results to return
        kind_filter: Filter by media kind (image, video_frame, etc.)
        time_filter_hours: Only search within last N hours
        similarity_threshold: Minimum similarity score
    """
    # Build advanced query with time-series filtering
    base_sql = """
    SELECT 
        media_id, 
        kind, 
        metadata, 
        uploaded_at,
        1 - (embedding <=> %s::vector) AS similarity_score,
        embedding <=> %s::vector AS distance
    FROM media_embeddings
    WHERE 1 = 1
    """
    
    params = [vector, vector]
    
    # Add filters
    if kind_filter:
        base_sql += " AND kind = %s"
        params.append(kind_filter)
    
    if time_filter_hours:
        base_sql += " AND uploaded_at > NOW() - INTERVAL '%s hours'"
        params.append(time_filter_hours)
    
    # Add similarity threshold
    base_sql += " AND (1 - (embedding <=> %s::vector)) >= %s"
    params.extend([vector, similarity_threshold])
    
    # Order by similarity (distance) and limit results
    base_sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
    params.extend([vector, top_k])
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(base_sql, params)
                results = cur.fetchall()
        
        logger.debug("Advanced embedding search completed", 
                    results_count=len(results), 
                    top_k=top_k, 
                    kind_filter=kind_filter,
                    time_filter_hours=time_filter_hours,
                    similarity_threshold=similarity_threshold)
        return results
        
    except Exception as e:
        logger.error("Failed to search embeddings with Timescale AI", 
                    top_k=top_k, error=str(e))
        raise

def search_embedding(vector: List[float], top_k: int = 5, kind_filter: Optional[str] = None) -> List[Tuple]:
    """Legacy search function for backward compatibility."""
    results = search_embedding_with_timescale_ai(
        vector=vector,
        top_k=top_k,
        kind_filter=kind_filter,
        similarity_threshold=0.0  # No threshold for legacy compatibility
    )
    # Convert to legacy format (media_id, kind, metadata, distance)
    return [(r[0], r[1], r[2], r[5]) for r in results]

def get_embedding_by_media_id(media_id: str) -> Optional[Tuple]:
    """Get embedding for a specific media ID."""
    sql = "SELECT media_id, kind, embedding, metadata FROM media_embeddings WHERE media_id = %s"
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (media_id,))
                result = cur.fetchone()
        
        return result
        
    except Exception as e:
        logger.error("Failed to get embedding", media_id=media_id, error=str(e))
        raise

# Audio fingerprint management functions
def insert_fingerprint(fp_hash: str, media_id: str, offset_seconds: float, fingerprint_data: Optional[bytes] = None):
    """Insert a new audio fingerprint into the database."""
    sql = """
    INSERT INTO audio_fingerprints (fp_hash, media_id, offset_seconds, fingerprint_data)
    VALUES (%s, %s, %s, %s)
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (fp_hash, media_id, offset_seconds, fingerprint_data))
                conn.commit()
        
        logger.debug("Fingerprint inserted successfully", 
                    fp_hash=fp_hash, media_id=media_id, offset=offset_seconds)
                    
    except Exception as e:
        logger.error("Failed to insert fingerprint", 
                    fp_hash=fp_hash, media_id=media_id, error=str(e))
        raise

def search_fingerprint(fp_hash: str) -> List[Tuple]:
    """Search for matching fingerprints."""
    sql = "SELECT media_id, offset_seconds FROM audio_fingerprints WHERE fp_hash = %s"
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (fp_hash,))
                results = cur.fetchall()
        
        logger.debug("Fingerprint search completed", 
                    fp_hash=fp_hash, results_count=len(results))
        return results
        
    except Exception as e:
        logger.error("Failed to search fingerprint", fp_hash=fp_hash, error=str(e))
        raise

# Image perceptual hash management functions
def insert_image_hashes(media_id: str, hashes: dict) -> None:
    """Insert image perceptual hashes into database."""
    sql = """
    INSERT INTO image_hashes (media_id, dhash, phash, ahash, dhash_16, phash_16)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (media_id) DO UPDATE SET
        dhash = EXCLUDED.dhash,
        phash = EXCLUDED.phash,
        ahash = EXCLUDED.ahash,
        dhash_16 = EXCLUDED.dhash_16,
        phash_16 = EXCLUDED.phash_16,
        updated_at = NOW()
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    media_id,
                    hashes.get('dhash'),
                    hashes.get('phash'),
                    hashes.get('ahash'), 
                    hashes.get('dhash_16'),
                    hashes.get('phash_16')
                ))
                conn.commit()
        
        logger.debug("Image hashes inserted", 
                    media_id=media_id, hash_types=list(hashes.keys()))
                    
    except Exception as e:
        logger.error("Failed to insert image hashes", 
                    media_id=media_id, error=str(e))
        raise

def search_similar_image_hashes(hashes: dict, exclude_media_id: Optional[str] = None) -> List[Tuple]:
    """Search for similar image hashes in database."""
    # Search for exact matches first (fastest)
    sql = """
    SELECT media_id, dhash, phash, ahash, dhash_16, phash_16, created_at
    FROM image_hashes 
    WHERE (dhash = %s OR phash = %s OR ahash = %s OR dhash_16 = %s OR phash_16 = %s)
    """
    params = [
        hashes.get('dhash'),
        hashes.get('phash'),
        hashes.get('ahash'),
        hashes.get('dhash_16'),
        hashes.get('phash_16')
    ]
    
    if exclude_media_id:
        sql += " AND media_id != %s"
        params.append(exclude_media_id)
    
    sql += " ORDER BY created_at DESC"
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                results = cur.fetchall()
        
        logger.debug("Image hash search completed", 
                    results_count=len(results),
                    hash_types=list(hashes.keys()))
        return results
        
    except Exception as e:
        logger.error("Failed to search image hashes", error=str(e))
        raise

def get_image_hashes(media_id: str) -> Optional[Dict]:
    """Get stored image hashes for a media file."""
    sql = """
    SELECT media_id, dhash, phash, ahash, dhash_16, phash_16, created_at, updated_at
    FROM image_hashes 
    WHERE media_id = %s
    """
    
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(sql, (media_id,))
                result = cur.fetchone()
        
        return dict(result) if result else None
        
    except Exception as e:
        logger.error("Failed to get image hashes", media_id=media_id, error=str(e))
        raise

# Media file management functions
def insert_media_file(
    media_id: str,
    filename: str,
    original_filename: str,
    content_type: str,
    file_size: int,
    file_hash: str,
    upload_user_id: Optional[str] = None,
    upload_ip: Optional[str] = None,
    status: str = "processing"
):
    """Insert a new media file record."""
    sql = """
    INSERT INTO media_files (
        media_id, filename, original_filename, content_type, file_size, 
        file_hash, upload_user_id, upload_ip, status
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    media_id, filename, original_filename, content_type, 
                    file_size, file_hash, upload_user_id, upload_ip, status
                ))
                conn.commit()
        
        logger.info("Media file record inserted", 
                   media_id=media_id, filename=filename, file_size=file_size)
                   
    except Exception as e:
        logger.error("Failed to insert media file record", 
                    media_id=media_id, filename=filename, error=str(e))
        raise

def update_media_file_status(
    media_id: str, 
    status: str, 
    storage_uri: Optional[str] = None,
    processing_results: Optional[Dict] = None
):
    """Update media file status and related information with normalized columns."""
    # Extract key fields from processing_results for dedicated columns
    matches_found = processing_results.get('matches_found', 0) if processing_results else 0
    processing_time_ms = processing_results.get('processing_time_ms', 0.0) if processing_results else 0.0
    media_type = processing_results.get('media_type') if processing_results else None
    
    sql = """
    UPDATE media_files 
    SET status = %s, storage_uri = COALESCE(%s, storage_uri), 
        processing_results = COALESCE(%s, processing_results),
        matches_found = %s, processing_time_ms = %s, media_type = %s,
        updated_at = NOW()
    WHERE media_id = %s
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    status, 
                    storage_uri, 
                    extras.Json(processing_results) if processing_results else None,
                    matches_found, processing_time_ms, media_type,
                    media_id
                ))
                conn.commit()
        
        logger.debug("Media file status updated with normalized data", 
                    media_id=media_id, status=status, storage_uri=storage_uri,
                    matches_found=matches_found, media_type=media_type)
                    
    except Exception as e:
        logger.error("Failed to update media file status", 
                    media_id=media_id, status=status, error=str(e))
        raise

def get_media_file(media_id: str) -> Optional[Dict]:
    """Get media file information by ID."""
    sql = """
    SELECT media_id, filename, original_filename, content_type, file_size,
           storage_uri, file_hash, status, processing_results, created_at, updated_at
    FROM media_files 
    WHERE media_id = %s
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(sql, (media_id,))
                result = cur.fetchone()
        
        return dict(result) if result else None
        
    except Exception as e:
        logger.error("Failed to get media file", media_id=media_id, error=str(e))
        raise

def find_duplicate_files(file_hash: str, exclude_media_id: Optional[str] = None) -> List[Dict]:
    """Find duplicate files by hash."""
    sql = """
    SELECT media_id, filename, content_type, file_size, created_at
    FROM media_files 
    WHERE file_hash = %s
    """
    params = [file_hash]
    
    if exclude_media_id:
        sql += " AND media_id != %s"
        params.append(exclude_media_id)
    
    sql += " ORDER BY created_at"
    
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                results = cur.fetchall()
        
        return [dict(row) for row in results]
        
    except Exception as e:
        logger.error("Failed to find duplicate files", file_hash=file_hash, error=str(e))
        raise

# Similarity match management functions
def insert_similarity_match(
    query_media_id: str,
    match_media_id: str,
    match_type: str,
    similarity_score: float,
    confidence_level: str = "medium",
    match_details: Optional[Dict] = None
):
    """Insert a similarity match record with normalized columns."""
    # Extract key fields from match_details for dedicated columns
    match_category = match_details.get('match_category') if match_details else None
    embedding_type = match_details.get('embedding_type') if match_details else None
    fingerprint_hash = match_details.get('fingerprint_hash') if match_details else None
    offset_seconds = match_details.get('offset') if match_details else None
    
    sql = """
    INSERT INTO similarity_matches (
        query_media_id, match_media_id, match_type, 
        similarity_score, confidence_level, match_details,
        match_category, embedding_type, fingerprint_hash, offset_seconds
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    query_media_id, match_media_id, match_type,
                    similarity_score, confidence_level, 
                    extras.Json(match_details) if match_details else None,
                    match_category, embedding_type, fingerprint_hash, offset_seconds
                ))
                conn.commit()
        
        logger.debug("Similarity match inserted with normalized data", 
                    query_media_id=query_media_id, match_media_id=match_media_id,
                    similarity_score=similarity_score, match_category=match_category,
                    embedding_type=embedding_type)
                    
    except Exception as e:
        logger.error("Failed to insert similarity match", 
                    query_media_id=query_media_id, match_media_id=match_media_id, error=str(e))
        raise

def get_similarity_matches(
    media_id: str, 
    limit: int = 10,
    min_score: float = 0.0,
    match_type_filter: Optional[str] = None
) -> List[Dict]:
    """Get similarity matches for a media file."""
    sql = """
    SELECT sm.match_media_id, sm.match_type, sm.similarity_score, 
           sm.confidence_level, sm.match_details, sm.created_at,
           mf.filename, mf.content_type, mf.storage_uri
    FROM similarity_matches sm
    LEFT JOIN media_files mf ON sm.match_media_id = mf.media_id
    WHERE sm.query_media_id = %s AND sm.similarity_score >= %s
    """
    params = [media_id, min_score]
    
    if match_type_filter:
        sql += " AND sm.match_type = %s"
        params.append(match_type_filter)
    
    sql += " ORDER BY sm.similarity_score DESC LIMIT %s"
    params.append(limit)
    
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                results = cur.fetchall()
        
        return [dict(row) for row in results]
        
    except Exception as e:
        logger.error("Failed to get similarity matches", 
                    media_id=media_id, error=str(e))
        raise

# Blockchain attribution functions
def insert_attribution_record(
    media_id: str,
    attribution_type: str = "original",
    blockchain_tx_hash: Optional[str] = None,
    blockchain_address: Optional[str] = None,
    proof_data: Optional[Dict] = None
):
    """Insert a blockchain attribution record with normalized columns."""
    # Extract key fields from proof_data for dedicated columns
    file_hash = proof_data.get('file_hash') if proof_data else None
    media_type = proof_data.get('media_type') if proof_data else None
    matches_found = proof_data.get('matches_found', 0) if proof_data else 0
    high_confidence_matches = proof_data.get('high_confidence_matches', 0) if proof_data else 0
    max_similarity_score = proof_data.get('max_similarity_score', 0.0) if proof_data else 0.0
    processing_time_ms = proof_data.get('processing_time_ms', 0.0) if proof_data else 0.0
    
    sql = """
    INSERT INTO attribution_records (
        media_id, blockchain_tx_hash, blockchain_address, 
        attribution_type, proof_data,
        file_hash, media_type, matches_found, high_confidence_matches,
        max_similarity_score, processing_time_ms
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    media_id, blockchain_tx_hash, blockchain_address,
                    attribution_type, extras.Json(proof_data) if proof_data else None,
                    file_hash, media_type, matches_found, high_confidence_matches,
                    max_similarity_score, processing_time_ms
                ))
                conn.commit()
        
        logger.info("Attribution record inserted with normalized data", 
                   media_id=media_id, attribution_type=attribution_type,
                   matches_found=matches_found, media_type=media_type)
                   
    except Exception as e:
        logger.error("Failed to insert attribution record", 
                    media_id=media_id, error=str(e))
        raise

def get_attribution_records(media_id: str) -> List[Dict]:
    """Get attribution records for a media file."""
    sql = """
    SELECT blockchain_tx_hash, blockchain_address, attribution_type, 
           proof_data, created_at
    FROM attribution_records 
    WHERE media_id = %s
    ORDER BY created_at DESC
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(sql, (media_id,))
                results = cur.fetchall()
        
        return [dict(row) for row in results]
        
    except Exception as e:
        logger.error("Failed to get attribution records", 
                    media_id=media_id, error=str(e))
        raise

# Timescale Vector AI specific functions
def create_vector_index_if_not_exists():
    """Create vector index for optimal similarity search performance."""
    sql = """
    CREATE INDEX IF NOT EXISTS idx_media_embeddings_vector_hnsw 
    ON media_embeddings USING hnsw (embedding vector_cosine_ops) 
    WITH (m = 16, ef_construction = 64);
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                conn.commit()
        
        logger.info("Vector HNSW index ensured")
        
    except Exception as e:
        logger.error("Failed to create vector index", error=str(e))
        raise

def get_vector_index_stats() -> Dict:
    """Get statistics about vector index performance."""
    sql = """
    SELECT 
        schemaname,
        tablename,
        indexname,
        idx_scan,
        idx_tup_read,
        idx_tup_fetch
    FROM pg_stat_user_indexes 
    WHERE indexname LIKE '%vector%';
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(sql)
                results = cur.fetchall()
        
        return [dict(row) for row in results]
        
    except Exception as e:
        logger.error("Failed to get vector index stats", error=str(e))
        return []

# Database utility functions
def check_database_connection() -> bool:
    """Check if Timescale database connection is working."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
        
        logger.info("Timescale database connection check successful")
        return result[0] == 1
        
    except Exception as e:
        logger.error("Timescale database connection check failed", error=str(e))
        return False

def get_database_stats() -> Dict:
    """Get comprehensive database statistics."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Get table counts
                cur.execute("""
                    SELECT 
                        (SELECT COUNT(*) FROM media_embeddings) as embeddings_count,
                        (SELECT COUNT(*) FROM audio_fingerprints) as fingerprints_count,
                        (SELECT COUNT(*) FROM media_files) as media_files_count,
                        (SELECT COUNT(*) FROM similarity_matches) as matches_count,
                        (SELECT COUNT(*) FROM attribution_records) as attribution_count,
                        (SELECT version()) as postgres_version,
                        (SELECT extversion FROM pg_extension WHERE extname = 'vector') as vector_version
                """)
                
                result = cur.fetchone()
                
                return {
                    "embeddings_count": result[0],
                    "fingerprints_count": result[1],
                    "media_files_count": result[2],
                    "similarity_matches_count": result[3],
                    "attribution_records_count": result[4],
                    "postgres_version": result[5],
                    "vector_extension_version": result[6],
                    "connection_pool_size": len(_connection_pool._pool) if _connection_pool else 0,
                    "database_type": "timescale"
                }
                
    except Exception as e:
        logger.error("Failed to get database stats", error=str(e))
        return {"error": str(e)}

def cleanup_old_records(days_old: int = 30) -> Dict:
    """Clean up old temporary records (for maintenance)."""
    try:
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Clean up failed media files
                cur.execute("""
                    DELETE FROM media_files 
                    WHERE status = 'failed' AND created_at < %s
                """, (cutoff_date,))
                failed_files_deleted = cur.rowcount
                
                # Clean up orphaned embeddings (no corresponding media file)
                cur.execute("""
                    DELETE FROM media_embeddings 
                    WHERE NOT EXISTS (
                        SELECT 1 FROM media_files 
                        WHERE media_files.media_id = media_embeddings.media_id
                    ) AND created_at < %s
                """, (cutoff_date,))
                orphaned_embeddings_deleted = cur.rowcount
                
                conn.commit()
        
        result = {
            "failed_files_deleted": failed_files_deleted,
            "orphaned_embeddings_deleted": orphaned_embeddings_deleted,
            "cutoff_date": cutoff_date.isoformat()
        }
        
        logger.info("Database cleanup completed", **result)
        return result
        
    except Exception as e:
        logger.error("Database cleanup failed", error=str(e))
        raise

# Initialize connection pool on module import
try:
    initialize_connection_pool()
    # Ensure vector index exists
    create_vector_index_if_not_exists()
except Exception as e:
    logger.warning("Could not initialize Timescale database at startup", error=str(e)) 