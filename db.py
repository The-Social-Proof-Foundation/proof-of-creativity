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

# Database connection configuration
DB_DSN = os.getenv("DB_DSN", "postgresql://user:password@localhost:5432/proof_of_creativity")
MIN_CONNECTIONS = 1
MAX_CONNECTIONS = 20

# Global connection pool
_connection_pool = None

def initialize_connection_pool():
    """Initialize the database connection pool."""
    global _connection_pool
    if _connection_pool is None:
        try:
            _connection_pool = SimpleConnectionPool(
                MIN_CONNECTIONS, 
                MAX_CONNECTIONS, 
                DB_DSN
            )
            logger.info("Database connection pool initialized", 
                       min_connections=MIN_CONNECTIONS, 
                       max_connections=MAX_CONNECTIONS)
        except Exception as e:
            logger.error("Failed to initialize database connection pool", error=str(e))
            raise

@contextmanager
def get_db_connection():
    """Context manager for database connections with automatic cleanup."""
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
    return psycopg2.connect(DB_DSN)

# Embedding management functions
def insert_embedding(media_id: str, kind: str, embedding: List[float], metadata: Dict[str, Any]):
    """Insert a new embedding into the database."""
    sql = """
    INSERT INTO media_embeddings (media_id, kind, embedding, metadata)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (media_id) DO UPDATE SET
        embedding = EXCLUDED.embedding,
        metadata = EXCLUDED.metadata,
        updated_at = NOW()
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (media_id, kind, embedding, extras.Json(metadata)))
                conn.commit()
        
        logger.debug("Embedding inserted successfully", 
                    media_id=media_id, kind=kind, metadata=metadata)
                    
    except Exception as e:
        logger.error("Failed to insert embedding", 
                    media_id=media_id, kind=kind, error=str(e))
        raise

def search_embedding(vector: List[float], top_k: int = 5, kind_filter: Optional[str] = None) -> List[Tuple]:
    """Search for similar embeddings using vector similarity."""
    # Build the SQL query with optional kind filter
    base_sql = """
    SELECT media_id, kind, metadata, embedding <-> %s AS dist
    FROM media_embeddings
    """
    
    params = [vector]
    
    if kind_filter:
        base_sql += " WHERE kind = %s"
        params.append(kind_filter)
    
    base_sql += " ORDER BY dist LIMIT %s"
    params.append(top_k)
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(base_sql, params)
                results = cur.fetchall()
        
        logger.debug("Embedding search completed", 
                    results_count=len(results), top_k=top_k, kind_filter=kind_filter)
        return results
        
    except Exception as e:
        logger.error("Failed to search embeddings", 
                    top_k=top_k, kind_filter=kind_filter, error=str(e))
        raise

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
    ON CONFLICT (fp_hash, media_id) DO UPDATE SET
        offset_seconds = EXCLUDED.offset_seconds,
        fingerprint_data = EXCLUDED.fingerprint_data,
        updated_at = NOW()
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
    ON CONFLICT (media_id) DO UPDATE SET
        filename = EXCLUDED.filename,
        original_filename = EXCLUDED.original_filename,
        content_type = EXCLUDED.content_type,
        file_size = EXCLUDED.file_size,
        file_hash = EXCLUDED.file_hash,
        status = EXCLUDED.status,
        updated_at = NOW()
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
    """Update media file status and related information."""
    sql = """
    UPDATE media_files 
    SET status = %s, storage_uri = COALESCE(%s, storage_uri), 
        processing_results = COALESCE(%s, processing_results), updated_at = NOW()
    WHERE media_id = %s
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    status, 
                    storage_uri, 
                    extras.Json(processing_results) if processing_results else None,
                    media_id
                ))
                conn.commit()
        
        logger.debug("Media file status updated", 
                    media_id=media_id, status=status, storage_uri=storage_uri)
                    
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
    """Insert a similarity match record."""
    sql = """
    INSERT INTO similarity_matches (
        query_media_id, match_media_id, match_type, 
        similarity_score, confidence_level, match_details
    )
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    query_media_id, match_media_id, match_type,
                    similarity_score, confidence_level, 
                    extras.Json(match_details) if match_details else None
                ))
                conn.commit()
        
        logger.debug("Similarity match inserted", 
                    query_media_id=query_media_id, match_media_id=match_media_id,
                    similarity_score=similarity_score)
                    
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
    blockchain_tx_hash: Optional[str] = None,
    blockchain_address: Optional[str] = None,
    attribution_type: str = "original",
    proof_data: Optional[Dict] = None
):
    """Insert a blockchain attribution record."""
    sql = """
    INSERT INTO attribution_records (
        media_id, blockchain_tx_hash, blockchain_address, 
        attribution_type, proof_data
    )
    VALUES (%s, %s, %s, %s, %s)
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    media_id, blockchain_tx_hash, blockchain_address,
                    attribution_type, extras.Json(proof_data) if proof_data else None
                ))
                conn.commit()
        
        logger.info("Attribution record inserted", 
                   media_id=media_id, attribution_type=attribution_type)
                   
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

# Database utility functions
def check_database_connection() -> bool:
    """Check if database connection is working."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
        
        logger.info("Database connection check successful")
        return result[0] == 1
        
    except Exception as e:
        logger.error("Database connection check failed", error=str(e))
        return False

def get_database_stats() -> Dict:
    """Get database statistics."""
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
                        (SELECT COUNT(*) FROM attribution_records) as attribution_count
                """)
                
                result = cur.fetchone()
                
                return {
                    "embeddings_count": result[0],
                    "fingerprints_count": result[1],
                    "media_files_count": result[2],
                    "similarity_matches_count": result[3],
                    "attribution_records_count": result[4],
                    "connection_pool_size": len(_connection_pool._pool) if _connection_pool else 0
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
except Exception as e:
    logger.warning("Could not initialize connection pool at startup", error=str(e))
