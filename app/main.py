import os
import mimetypes
import hashlib
import structlog
import time
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import our modules with new structure
from app import config
from app.services import embedding, fingerprint, video_processing, image_hash
from app.core.storage import StorageClient
from app.core.database import (
    insert_embedding, search_embedding_with_timescale_ai, insert_fingerprint, search_fingerprint,
    insert_image_hashes, search_similar_image_hashes, get_image_hashes,
    insert_media_file, update_media_file_status, insert_similarity_match, get_similarity_matches,
    insert_attribution_record, get_attribution_records, get_database_stats,
    check_database_connection
)
from app.core.utils import save_temp_upload, new_media_id, calculate_file_hash, cleanup_temp_file
from app.models.similarity import MediaMatch, UploadResponse, ErrorResponse, HealthResponse

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Global storage client
storage_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    global storage_client
    
    # Startup
    logger.info("Starting Proof of Creativity API")
    try:
        storage_client = StorageClient()
        logger.info("Storage client initialized", storage_backend="GCS" if config.USE_GCS else "Walrus")
        
        # Test database connection
        if check_database_connection():
            logger.info("Timescale database connection verified")
        else:
            logger.warning("Database connection check failed")
            
    except Exception as e:
        logger.error("Failed to initialize application", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Proof of Creativity API")

# Create FastAPI application
app = FastAPI(
    title="Proof of Creativity API",
    description="Scalable Media Attribution Architecture for detecting original vs derivative media using Timescale Vector AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    responses={
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    }
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration constants
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 100 * 1024 * 1024))  # 100MB default
SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
SUPPORTED_AUDIO_TYPES = {"audio/mpeg", "audio/wav", "audio/flac", "audio/ogg"}
SUPPORTED_VIDEO_TYPES = {"video/mp4", "video/avi", "video/mov", "video/webm"}
SUPPORTED_TYPES = SUPPORTED_IMAGE_TYPES | SUPPORTED_AUDIO_TYPES | SUPPORTED_VIDEO_TYPES

async def validate_file(file: UploadFile) -> tuple[str, str]:
    """Validate uploaded file and return content type and media type."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required"
        )
    
    # Check file size
    if hasattr(file, 'size') and file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum allowed size of {MAX_FILE_SIZE} bytes"
        )
    
    # Determine content type
    content_type = file.content_type or mimetypes.guess_type(file.filename)[0]
    if not content_type:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not determine file type"
        )
    
    if content_type not in SUPPORTED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported media type: {content_type}. Supported types: {', '.join(SUPPORTED_TYPES)}"
        )
    
    # Determine media category
    if content_type in SUPPORTED_IMAGE_TYPES:
        media_type = "image"
    elif content_type in SUPPORTED_AUDIO_TYPES:
        media_type = "audio"
    elif content_type in SUPPORTED_VIDEO_TYPES:
        media_type = "video"
    else:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported media type: {content_type}"
        )
    
    return content_type, media_type

async def detect_image_similarity(file_path: str, media_id: str) -> List[MediaMatch]:
    """Process image and detect similar content using dual-mode approach: perceptual hashing + CLIP embeddings."""
    try:
        logger.info("Processing image for similarity detection", media_id=media_id)
        
        # Step 1: Generate and check perceptual hashes for exact/near-exact duplicates
        logger.debug("Generating perceptual hashes", media_id=media_id)
        image_hashes = image_hash.generate_image_hashes(file_path)
        
        # Search for exact hash matches (true duplicates)
        hash_matches = search_similar_image_hashes(image_hashes, exclude_media_id=media_id)
        
        # Store image hashes
        insert_image_hashes(media_id, image_hashes)
        
        matches = []
        
        # Process perceptual hash matches (highest priority - exact duplicates)
        for hash_match in hash_matches:
            match_media_id = hash_match[0]  # First column is media_id
            stored_hashes = {
                'dhash': hash_match[1],
                'phash': hash_match[2], 
                'ahash': hash_match[3],
                'dhash_16': hash_match[4],
                'phash_16': hash_match[5]
            }
            
            # Calculate exact similarity between hashes
            hash_similarity, match_type = image_hash.calculate_hash_similarity(image_hashes, stored_hashes)
            
            if hash_similarity >= 0.99:  # 99%+ hash similarity = exact duplicate
                confidence = "high"
                match_category = "exact_duplicate"
                
                # Record the perceptual hash match
                insert_similarity_match(
                    query_media_id=media_id,
                    match_media_id=match_media_id,
                    match_type="perceptual_hash",
                    similarity_score=float(hash_similarity),
                    confidence_level=confidence,
                    match_details={
                        "hash_type": match_type,
                        "match_category": match_category,
                        "hash_similarity": hash_similarity
                    }
                )
                
                matches.append(MediaMatch(
                    media_id=match_media_id,
                    similarity_score=float(hash_similarity),
                    match_type="perceptual_hash",
                    confidence_level=confidence
                ))
        
        # Step 2: Only check CLIP embeddings if no perceptual hash duplicates found
        if not matches:
            logger.debug("No perceptual hash duplicates found, checking CLIP embeddings", media_id=media_id)
            
            # Generate CLIP embedding
            embedding_vector = embedding.image_embedding(file_path)
            
            # Use higher threshold for semantic similarity (98% for near-duplicates, 90% for similar content)
            duplicate_threshold = 0.98  # Near-duplicate via semantic similarity (raised from 0.95)
            similar_threshold = 0.90    # Similar content (raised from 0.85)
            
            # Search for similar embeddings
            duplicate_embeddings = search_embedding_with_timescale_ai(
                vector=embedding_vector, 
                top_k=10,
                kind_filter="image",
                similarity_threshold=duplicate_threshold
            )
            
            # Only check for similar content if no duplicates found
            similar_embeddings = search_embedding_with_timescale_ai(
                vector=embedding_vector, 
                top_k=10,
                kind_filter="image", 
                similarity_threshold=similar_threshold
            ) if not duplicate_embeddings else []
            
            # Store the new embedding
            insert_embedding(media_id, "image", embedding_vector, {})
            
            # Process CLIP embedding matches
            all_embeddings = duplicate_embeddings + similar_embeddings
            
            for match in all_embeddings:
                match_media_id, kind, metadata, uploaded_at, similarity_score, distance = match
                if match_media_id != media_id:  # Don't match against self
                    
                    # Categorize based on CLIP similarity score (stricter thresholds)
                    if similarity_score >= 0.99:
                        confidence = "high"
                        match_category = "semantic_duplicate"
                    elif similarity_score >= 0.98:
                        confidence = "high" 
                        match_category = "semantic_near_duplicate"
                    elif similarity_score >= 0.95:
                        confidence = "medium"
                        match_category = "similar_content"
                    elif similarity_score >= 0.90:
                        confidence = "medium"
                        match_category = "related_content"
                    else:
                        confidence = "low"
                        match_category = "loosely_related"
                    
                    # Record the CLIP embedding match
                    insert_similarity_match(
                        query_media_id=media_id,
                        match_media_id=match_media_id,
                        match_type="embedding",
                        similarity_score=float(similarity_score),
                        confidence_level=confidence,
                        match_details={
                            "embedding_type": "clip", 
                            "kind": kind,
                            "match_category": match_category,
                            "duplicate_threshold": duplicate_threshold,
                            "similar_threshold": similar_threshold
                        }
                    )
                    
                    matches.append(MediaMatch(
                        media_id=match_media_id,
                        similarity_score=float(similarity_score),
                        match_type="embedding",
                        confidence_level=confidence
                    ))
            
            logger.info("CLIP embedding search completed", 
                       media_id=media_id, 
                       duplicates_found=len(duplicate_embeddings),
                       similar_found=len(similar_embeddings))
        else:
            # Still store CLIP embedding for future searches, but don't search with it
            logger.debug("Perceptual duplicates found, storing CLIP embedding without search", media_id=media_id)
            embedding_vector = embedding.image_embedding(file_path)
            insert_embedding(media_id, "image", embedding_vector, {})
        
        logger.info("Image similarity detection completed", 
                   media_id=media_id, 
                   matches_found=len(matches),
                   hash_matches_found=len(hash_matches))
        return matches
        
    except Exception as e:
        logger.error("Failed to detect image similarity", 
                    media_id=media_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Image similarity detection failed: {str(e)}")

async def detect_audio_similarity(file_path: str, media_id: str) -> List[MediaMatch]:
    """Process audio and detect similar content."""
    try:
        logger.info("Processing audio for similarity detection", media_id=media_id)
        
        # Generate fingerprint
        fp_hash = fingerprint.fingerprint_audio(file_path)
        
        # Search for similar fingerprints
        similar_fingerprints = search_fingerprint(fp_hash)
        
        # Store the new fingerprint
        insert_fingerprint(fp_hash, media_id, 0.0)
        
        # Process matches
        matches = []
        for match in similar_fingerprints:
            match_media_id, offset = match
            if match_media_id != media_id:  # Don't match against self
                # For fingerprints, exact match means high confidence
                confidence = "high"
                similarity_score = 1.0  # Exact fingerprint match
                
                # Record the match in database
                insert_similarity_match(
                    query_media_id=media_id,
                    match_media_id=match_media_id,
                    match_type="fingerprint",
                    similarity_score=similarity_score,
                    confidence_level=confidence,
                    match_details={"fingerprint_hash": fp_hash, "offset": offset}
                )
                
                matches.append(MediaMatch(
                    media_id=match_media_id,
                    similarity_score=similarity_score,
                    match_type="fingerprint",
                    confidence_level=confidence
                ))
        
        logger.info("Audio similarity detection completed", 
                   media_id=media_id, matches_found=len(matches))
        return matches
        
    except Exception as e:
        logger.error("Error in audio similarity detection", 
                    media_id=media_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing audio: {str(e)}"
        )

async def detect_video_similarity(file_path: str, media_id: str) -> List[MediaMatch]:
    """Process video and detect similar content."""
    try:
        logger.info("Processing video for similarity detection", media_id=media_id)
        
        # Process video (extract frames and audio)
        frame_embeddings, audio_fp_hash = video_processing.process_video(file_path)
        
        all_matches = []
        
        # Process frame embeddings using Timescale Vector AI
        for idx, frame_embedding in enumerate(frame_embeddings):
            similar_embeddings = search_embedding_with_timescale_ai(
                vector=frame_embedding, 
                top_k=5,
                kind_filter="video_frame",
                similarity_threshold=0.7
            )
            frame_media_id = f"{media_id}_frame_{idx}"
            
            # Store frame embedding
            insert_embedding(
                frame_media_id, 
                "video_frame", 
                frame_embedding, 
                {"parent_media_id": media_id, "frame_index": idx}
            )
            
            # Process frame matches
            for match in similar_embeddings:
                match_media_id, kind, metadata, uploaded_at, similarity_score, distance = match
                if not match_media_id.startswith(media_id):  # Don't match against self
                    confidence = "high" if similarity_score > 0.9 else "medium" if similarity_score > 0.7 else "low"
                    
                    insert_similarity_match(
                        query_media_id=media_id,
                        match_media_id=match_media_id,
                        match_type="embedding",
                        similarity_score=float(similarity_score),
                        confidence_level=confidence,
                        match_details={"embedding_type": "clip", "kind": kind, "frame_index": idx}
                    )
                    
                    all_matches.append(MediaMatch(
                        media_id=match_media_id,
                        similarity_score=float(similarity_score),
                        match_type="embedding",
                        confidence_level=confidence
                    ))
        
        # Process audio fingerprint
        if audio_fp_hash:
            similar_fingerprints = search_fingerprint(audio_fp_hash)
            insert_fingerprint(audio_fp_hash, media_id, 0.0)
            
            for match in similar_fingerprints:
                match_media_id, offset = match
                if match_media_id != media_id:
                    confidence = "high"
                    similarity_score = 1.0
                    
                    insert_similarity_match(
                        query_media_id=media_id,
                        match_media_id=match_media_id,
                        match_type="fingerprint",
                        similarity_score=similarity_score,
                        confidence_level=confidence,
                        match_details={"fingerprint_hash": audio_fp_hash, "offset": offset}
                    )
                    
                    all_matches.append(MediaMatch(
                        media_id=match_media_id,
                        similarity_score=similarity_score,
                        match_type="fingerprint",
                        confidence_level=confidence
                    ))
        
        # Deduplicate matches by media_id (keep highest score)
        unique_matches = {}
        for match in all_matches:
            if match.media_id not in unique_matches or match.similarity_score > unique_matches[match.media_id].similarity_score:
                unique_matches[match.media_id] = match
        
        final_matches = list(unique_matches.values())
        logger.info("Video similarity detection completed", 
                   media_id=media_id, matches_found=len(final_matches))
        return final_matches
        
    except Exception as e:
        logger.error("Error in video similarity detection", 
                    media_id=media_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing video: {str(e)}"
        )

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Proof of Creativity API",
        "version": "1.0.0",
        "description": "Scalable Media Attribution Architecture with Timescale Vector AI",
        "docs_url": "/docs",
        "health_url": "/health",
        "database": "Timescale with Vector AI"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with comprehensive system status."""
    try:
        # Check database
        db_healthy = check_database_connection()
        db_stats = get_database_stats() if db_healthy else {}
        
        # Check storage
        storage_health = storage_client.health_check() if storage_client else {"error": "not_initialized"}
        
        components = {
            "database": "healthy" if db_healthy else "unhealthy",
            "storage": "healthy" if storage_health.get("gcs", {}).get("available") or storage_health.get("walrus", {}).get("available") else "unhealthy",
            "embedding_model": "healthy",  # Could add actual model health checks
            "fingerprint_service": "healthy",
            "timescale_vector": "healthy" if db_stats.get("vector_extension_version") else "unavailable"
        }
        
        overall_status = "healthy" if all(status == "healthy" for status in components.values()) else "degraded"
        
        return HealthResponse(
            status=overall_status,
            version="1.0.0",
            components={
                **components,
                "database_stats": db_stats,
                "storage_health": storage_health
            }
        )
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return HealthResponse(
            status="unhealthy",
            version="1.0.0",
            components={"error": str(e)}
        )

@app.post("/upload", response_model=UploadResponse)
async def upload_media(
    file: UploadFile = File(..., description="Media file to upload and analyze")
):
    """
    Upload and analyze media file for similarity detection using Timescale Vector AI.
    
    Supports:
    - Images: JPEG, PNG, GIF, WebP
    - Audio: MP3, WAV, FLAC, OGG  
    - Video: MP4, AVI, MOV, WebM
    
    Returns similarity matches with other media in the database using advanced vector search.
    """
    media_id = new_media_id()
    temp_file_path = None
    start_time = time.time()
    
    try:
        logger.info("Processing media upload", 
                   media_id=media_id, filename=file.filename, content_type=file.content_type)
        
        # Validate file
        content_type, media_type = await validate_file(file)
        
        # Save temporary file
        temp_file_path = save_temp_upload(file)
        file_size = os.path.getsize(temp_file_path)
        
        # Calculate file hash for deduplication
        file_hash = calculate_file_hash(temp_file_path)
        
        # Insert media file record
        insert_media_file(
            media_id=media_id,
            filename=file.filename,
            original_filename=file.filename,
            content_type=content_type,
            file_size=file_size,
            file_hash=file_hash,
            status="processing"
        )
        
        # Detect similarity based on media type
        matches = []
        if media_type == "image":
            matches = await detect_image_similarity(temp_file_path, media_id)
        elif media_type == "audio":
            matches = await detect_audio_similarity(temp_file_path, media_id)
        elif media_type == "video":
            matches = await detect_video_similarity(temp_file_path, media_id)
        
        # Upload to storage
        with open(temp_file_path, "rb") as f:
            storage_uri = storage_client.upload(f"{media_id}_{file.filename}", f, media_type)
        
        # Update media file with storage URI and completion status
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        update_media_file_status(
            media_id=media_id,
            status="completed",
            storage_uri=storage_uri,
            processing_results={
                "processing_time_ms": processing_time,
                "matches_found": len(matches),
                "media_type": media_type
            }
        )
        
        # Record attribution automatically based on similarity analysis
        attribution_type = determine_attribution_type(matches)
        proof_data = {
            "file_hash": file_hash,
            "original_filename": file.filename,
            "upload_timestamp": time.time(),
            "processing_time_ms": processing_time,
            "matches_found": len(matches),
            "media_type": media_type,
            "high_confidence_matches": len([m for m in matches if m.confidence_level == "high"]),
            "match_types": list(set(m.match_type for m in matches)) if matches else [],
            "max_similarity_score": max([m.similarity_score for m in matches]) if matches else 0.0
        }
        
        insert_attribution_record(
            media_id=media_id,
            attribution_type=attribution_type,
            proof_data=proof_data
        )
        
        logger.info("Attribution recorded", 
                   media_id=media_id, 
                   attribution_type=attribution_type,
                   matches_found=len(matches))
        
        # Generate response message based on findings
        if matches:
            high_confidence_count = len([m for m in matches if m.confidence_level == "high"])
            if high_confidence_count > 0:
                message = f"Found {high_confidence_count} high-confidence matches - possible derivative content"
            else:
                message = f"Found {len(matches)} potential matches - review for similarity"
        else:
            message = "No similar content found - appears to be original"
        
        return UploadResponse(
            media_id=media_id,
            filename=file.filename,
            content_type=content_type,
            file_size=file_size,
            file_hash=file_hash,
            storage_uri=storage_uri,
            matches=matches,
            processing_status="completed",
            message=message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error in upload", 
                    media_id=media_id, error=str(e))
        
        # Update status to failed
        try:
            update_media_file_status(media_id, "failed")
        except:
            pass  # Don't let status update failures mask the original error
            
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal processing error: {str(e)}"
        )
    finally:
        # Cleanup temporary file
        if temp_file_path:
            cleanup_temp_file(temp_file_path)

def determine_attribution_type(matches: List[MediaMatch]) -> str:
    """
    Determine attribution type based on similarity analysis.
    
    Logic:
    - No matches: 'original' 
    - 1-2 high confidence matches: 'derivative'
    - 3+ high confidence matches: 'remix'
    - (Future: blockchain verified: 'licensed')
    """
    if not matches:
        return "original"
    
    high_confidence_matches = [m for m in matches if m.confidence_level == "high"]
    
    if len(high_confidence_matches) == 0:
        return "original"  # No high confidence matches
    elif len(high_confidence_matches) <= 2:
        return "derivative"  # Clear derivative from 1-2 sources
    else:
        return "remix"  # Multiple sources combined

@app.get("/media/{media_id}/matches", response_model=List[MediaMatch])
async def get_media_matches(
    media_id: str,
    limit: int = Query(default=10, ge=1, le=100, description="Maximum number of matches to return")
):
    """Get similarity matches for a specific media file."""
    try:
        matches = get_similarity_matches(media_id, limit=limit)
        return [
            MediaMatch(
                media_id=match["match_media_id"],
                similarity_score=match["similarity_score"],
                match_type=match["match_type"],
                confidence_level=match["confidence_level"]
            )
            for match in matches
        ]
    except Exception as e:
        logger.error("Failed to get media matches", media_id=media_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get matches: {str(e)}")

@app.get("/media/{media_id}/attribution", response_model=List[dict])
async def get_media_attribution(media_id: str):
    """Get attribution records for a specific media file."""
    try:
        attribution_records = get_attribution_records(media_id)
        return attribution_records
    except Exception as e:
        logger.error("Failed to get attribution records", media_id=media_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get attribution: {str(e)}")

@app.get("/stats", response_model=dict)
async def get_system_stats():
    """Get comprehensive system statistics."""
    try:
        db_stats = get_database_stats()
        storage_health = storage_client.health_check() if storage_client else {}
        
        return {
            "database": db_stats,
            "storage": storage_health,
            "api_version": "1.0.0",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error("Error retrieving system stats", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving stats: {str(e)}"
        )

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("Unhandled exception", 
                url=str(request.url), method=request.method, error=str(exc), exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "internal_server_error", "message": "An unexpected error occurred"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("DEBUG", "false").lower() == "true",
        log_config=None,  # We handle logging with structlog
    ) 