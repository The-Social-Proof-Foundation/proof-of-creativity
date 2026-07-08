import os
import mimetypes
import hashlib
import structlog
import time
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status, Query, Request, WebSocket, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import our modules with new structure
from app import config
from app.services.media_similarity import detect_image_similarity, detect_audio_similarity
from app.services.poc_video import analyze_video_similarity

# Older streaming handlers referenced this name; stale reloads or forks can still evaluate it at import/runtime.
detect_video_similarity = analyze_video_similarity

from app.services.poc_submission import (
    attempt_proof_of_creativity_submission,
    attribution_type_from_chain_summary,
    build_upload_attribution_message,
    preview_poc_chain_summary_for_upload,
    preview_video_track_attestation,
)
from app.services.poc_utils import (
    OFFCHAIN_DEFAULT_POC_CONFIG,
    mys_integration_enabled_from_env,
    mysocial_readiness_payload,
    poc_fail_lifespan_on_mys_misconfig_from_env,
    poc_require_tx_when_post_id_from_env,
    similarity_float_to_u64_percent,
)
from app.core.storage import StorageClient
from app.core.database import (
    insert_media_file,
    update_media_file_status,
    insert_attribution_record,
    get_attribution_records,
    get_similarity_matches,
    get_database_stats,
    check_database_connection,
)
from app.core.utils import save_temp_upload, new_media_id, calculate_file_hash, cleanup_temp_file
from app.models.similarity import MediaMatch, UploadResponse, ErrorResponse, HealthResponse, StreamingUploadResponse

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

# Global Redis cache
redis_cache = None

# Global MySocial client (optional)
myso_client = None

# Global progress tracker
progress_tracker = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    global storage_client, redis_cache, myso_client, progress_tracker
    
    # Startup
    logger.info(
        "Starting Proof of Creativity API",
        main_module_path=os.path.abspath(__file__),
    )
    try:
        # Step 1: Run database migrations first (like Diesel's embedded_migrations)
        logger.info("Running database migrations...")
        from app.core.migrations import run_migrations
        migrations_ok = run_migrations()
        
        if migrations_ok:
            logger.info("✅ Database migrations completed")
        else:
            logger.warning("⚠️  Database migrations failed - continuing anyway")
        
        # Step 2: Initialize Redis cache (for performance)
        from app.core.redis_client import get_redis
        redis_cache = get_redis()
        if redis_cache.enabled:
            logger.info("✅ Redis cache initialized")
        else:
            logger.warning("⚠️  Redis cache disabled - performance will be slower")
        
        # Step 2b: Initialize progress tracker (uses Redis)
        from app.core.progress_tracker import ProgressTracker
        progress_tracker = ProgressTracker(redis_cache)
        logger.info("✅ Progress tracker initialized")
        
        # Step 3: Initialize database connection pool (after migrations)
        from app.core.database import initialize_connection_pool, create_vector_index_if_not_exists
        initialize_connection_pool()
        logger.info("Database connection pool initialized")
        
        # Step 4: Create vector indexes (after tables exist)
        try:
            create_vector_index_if_not_exists()
            logger.info("Vector indexes ensured")
        except Exception as e:
            logger.warning("Could not create vector indexes", error=str(e))
        
        # Step 5: Initialize MySocial blockchain client (optional)
        from app.network_config import bootstrap_active_network_sessions
        from app.services.myso_client import init_myso_client

        bootstrap_active_network_sessions()
        myso_client = init_myso_client()
        if myso_client:
            logger.info("✅ MySocial blockchain integration enabled")
        else:
            logger.info("ℹ️  MySocial blockchain integration disabled or unavailable")
        if (
            mys_integration_enabled_from_env()
            and myso_client is None
            and poc_fail_lifespan_on_mys_misconfig_from_env()
        ):
            raise RuntimeError(
                "MySocial integration is enabled (MYSO_INTEGRATION_ENABLED) but the PoC client did not "
                "initialize (missing object IDs, MYSO_POC_STRICT_ORACLE mismatch, wallet error, etc.). "
                "Fix env or set MYSO_INTEGRATION_ENABLED=false. "
                "To allow boot without chain, unset MYSO_POC_FAIL_LIFESPAN_ON_MYS_MISCONFIG."
            )
        
        # Step 6: Initialize storage client
        storage_client = StorageClient()
        logger.info("Storage client initialized", 
                   storage_backend="R2" if config.USE_CLOUDFLARE_R2 else "GCS" if config.USE_GCS else "Local")
        
        # Step 7: Verify everything works
        if check_database_connection():
            logger.info("Database connection verified")
        else:
            logger.warning("Database connection check failed")

        # Oracle WebSocket event bus → hub fan-out
        from app.api.ws.hub import ws_hub
        from app.services.events import event_bus

        async def _forward_to_ws(event_type: str, message: dict) -> None:
            await ws_hub.broadcast(event_type, message)

        event_bus.subscribe_all(_forward_to_ws)
        logger.info("✅ Oracle WebSocket event bus wired")
            
    except Exception as e:
        logger.error("Failed to initialize application", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Proof of Creativity API")

# Create FastAPI application
app = FastAPI(
    title="Proof of Creativity API",
    description=(
        "Oracle service: media fingerprinting, similarity search (0–1 float scores + 0–100 integer percents for chain), "
        "and optional MySocial Move submission. See README for oracle vs indexer vs chain roles."
    ),
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

from app.api.rest.routes import router as oracle_rest_router
from app.api.rest.discovery_internal import router as discovery_internal_router
from app.api.ws.routes import router as oracle_ws_router

app.include_router(oracle_rest_router)
app.include_router(discovery_internal_router)
app.include_router(oracle_ws_router)

# Configuration constants
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 100 * 1024 * 1024))  # 100MB default
SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
SUPPORTED_AUDIO_TYPES = {"audio/mpeg", "audio/wav", "audio/flac", "audio/ogg"}
SUPPORTED_VIDEO_TYPES = {"video/mp4", "video/avi", "video/mov", "video/webm"}
SUPPORTED_TYPES = SUPPORTED_IMAGE_TYPES | SUPPORTED_AUDIO_TYPES | SUPPORTED_VIDEO_TYPES

async def check_rate_limit(request: Request):
    """Rate limiting dependency - configurable uploads per hour per IP"""
    if not redis_cache or not redis_cache.enabled:
        return  # No rate limiting if Redis disabled
    
    client_ip = request.client.host if request.client else "unknown"
    limit = config.RATE_LIMIT_UPLOADS_PER_HOUR
    
    allowed, current_count = redis_cache.check_rate_limit(
        identifier=client_ip,
        limit=limit,
        window=3600  # 1 hour
    )
    
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {limit} uploads per hour. Current: {current_count}"
        )

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

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Proof of Creativity API",
        "version": "1.0.0",
        "description": "Scalable Media Attribution Architecture with PostgreSQL + pgvector",
        "docs_url": "/docs",
        "health_url": "/health",
        "readyz_url": "/readyz",
        "database": "PostgreSQL with pgvector"
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
        
        # Check Redis
        redis_health = redis_cache.health_check() if redis_cache else {"available": False}
        
        components = {
            "database": "healthy" if db_healthy else "unhealthy",
            "storage": "healthy" if storage_health.get("r2", {}).get("available") or storage_health.get("gcs", {}).get("available") else "unhealthy",
            "redis": "healthy" if redis_health.get("available") else "unavailable",
            "embedding_model": "healthy",  # Could add actual model health checks
            "fingerprint_service": "healthy",
            "pgvector": "healthy" if db_stats.get("vector_extension_version") else "unavailable",
        }
        
        overall_status = "healthy" if all(status == "healthy" for status in components.values()) else "degraded"
        
        return HealthResponse(
            status=overall_status,
            version="1.0.0",
            components={
                **components,
                "database_stats": db_stats,
                "storage_health": storage_health,
                "redis_health": redis_health,
                "mysocial": mysocial_readiness_payload(myso_client),
            }
        )
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return HealthResponse(
            status="unhealthy",
            version="1.0.0",
            components={"error": str(e)}
        )

@app.get("/readyz")
async def readiness():
    """
    Load balancer / orchestrator probe: DB must be up; if MySocial integration is enabled,
    oracle client must be active and wallet must match PoCConfig oracle_address.
    """
    db_ok = check_database_connection()
    mys = mysocial_readiness_payload(myso_client)
    ready = db_ok and mys.get("ready_for_submission", True)
    payload = {
        "ready": ready,
        "database": "healthy" if db_ok else "unhealthy",
        "mysocial": mys,
    }
    return JSONResponse(
        status_code=200 if ready else 503,
        content=payload,
    )

@app.post("/upload/stream", response_model=StreamingUploadResponse)
async def upload_media_streaming(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Media file to upload and analyze"),
    post_id: Optional[str] = Query(None, description="MySocial post object id for PoC submission"),
    spt_pool_id: Optional[str] = Query(
        None, description="Optional Social Proof Token pool id (enables analyze_and_update_post_sync_token_pool)"
    ),
    force_reanalyze: bool = Query(
        False,
        description="Forwarded to post preflight; overturn-cleared posts omit active PoC without requiring this flag",
    ),
    creator_address: Optional[str] = Query(
        None, description="Optional creator wallet stored on this media_files row for attribution lookups",
    ),
    royalty_free: bool = Query(
        False,
        description="With post_id + MySocial: submit explicit royalty-free PoC outcome (on-chain outcome 4)",
    ),
    upload_to_stream: bool = False,
    _rate_limit: None = Depends(check_rate_limit),
):
    """
    Streaming upload with immediate response and WebSocket progress updates
    
    Returns predicted CDN URL immediately, processes in background
    
    Flow:
    1. Validate file
    2. Check post not already analyzed (if post_id provided)
    3. Generate media_id and predict URL
    4. Return immediately with WebSocket info
    5. Process upload + analysis in background
    
    Connect to WebSocket: ws://host:port/ws/upload/{upload_id} for real-time progress
    """
    from pathlib import Path
    
    try:
        # Validate file
        content_type, media_type = await validate_file(file)
        
        # Generate media_id upfront
        media_id = new_media_id()
        
        # Check if post was already analyzed (one media per post rule)
        if post_id and myso_client:
            try:
                post_check = myso_client.check_post_already_analyzed(post_id, force_reanalyze=force_reanalyze)
                if post_check.get("already_analyzed"):
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail=f"Post {post_id} was already analyzed by PoC. Only one media per post allowed."
                    )
            except HTTPException:
                raise
            except Exception as e:
                logger.warning("Could not check post status", post_id=post_id, error=str(e))
                # Continue anyway
        
        # Generate predicted URL (simplified: just media_id + extension)
        ext = Path(file.filename).suffix or ".bin"
        simplified_filename = f"{media_id}{ext}"
        
        # Construct predicted CDN URL
        if config.R2_PUBLIC_DOMAIN:
            from datetime import datetime
            now = datetime.now()
            year = now.strftime("%Y")
            month = now.strftime("%m")
            predicted_url = f"https://{config.R2_PUBLIC_DOMAIN}/{media_type}/{year}/{month}/{simplified_filename}"
        else:
            predicted_url = f"r2://{config.R2_BUCKET_NAME}/{media_type}/{simplified_filename}"
        
        # Save temp file for background processing
        temp_file_path = save_temp_upload(file)
        file_hash = calculate_file_hash(temp_file_path)
        file_size = os.path.getsize(temp_file_path)

        # Persist catalog row immediately (same as /upload). Background work may fail or be
        # interrupted (reload/worker); without this, streaming uploads leave no DB trace.
        try:
            insert_media_file(
                media_id=media_id,
                filename=simplified_filename,
                original_filename=file.filename or simplified_filename,
                content_type=content_type,
                file_size=file_size,
                file_hash=file_hash,
                status="processing",
                creator_address=creator_address,
            )
        except Exception as e:
            logger.error(
                "Failed to record streaming upload in database",
                media_id=media_id,
                error=str(e),
            )
            cleanup_temp_file(temp_file_path)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to record upload: {str(e)}",
            )

        # Create progress entry in Redis
        if progress_tracker:
            progress_tracker.create_upload(
                upload_id=media_id,
                predicted_url=predicted_url,
                post_id=post_id
            )
        
        # Spawn background task for processing
        from app.services.background_tasks import process_upload_background

        background_tasks.add_task(
            process_upload_background,
            upload_id=media_id,
            temp_file_path=temp_file_path,
            filename=file.filename,
            content_type=content_type,
            media_type=media_type,
            file_hash=file_hash,
            post_id=post_id,
            spt_pool_id=spt_pool_id,
            force_reanalyze=force_reanalyze,
            creator_address=creator_address,
            royalty_free=royalty_free,
            upload_to_stream=upload_to_stream,
            progress_tracker=progress_tracker,
            storage_client=storage_client,
            myso_client=myso_client,
        )
        
        # Return immediately with predicted URL and WebSocket info
        websocket_url = f"ws://{request.url.hostname}:{request.url.port}/ws/upload/{media_id}"
        
        logger.info("Upload accepted, processing in background",
                   upload_id=media_id,
                   post_id=post_id,
                   predicted_url=predicted_url)
        
        return StreamingUploadResponse(
            upload_id=media_id,
            predicted_url=predicted_url,
            websocket_url=websocket_url,
            message="Upload accepted, processing in background. Connect to WebSocket for real-time progress.",
            post_id=post_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Upload initiation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initiate upload: {str(e)}"
        )

@app.post("/upload", response_model=UploadResponse)
async def upload_media(
    request: Request,
    file: UploadFile = File(..., description="Media file to upload and analyze"),
    post_id: Optional[str] = Query(None, description="MySocial post object id"),
    spt_pool_id: Optional[str] = Query(None, description="Optional SPT pool id for synced PoC mint fields"),
    force_reanalyze: bool = Query(
        False,
        description="Preflight flag echoed to post lookups; overturn-cleared posts lack active PoC automatically",
    ),
    creator_address: Optional[str] = Query(
        None, description="Optional wallet stored alongside this upload for attribution mapping",
    ),
    royalty_free: bool = Query(
        False,
        description="With post_id + MySocial: submit explicit royalty-free PoC outcome (on-chain outcome 4)",
    ),
    e2e_score: Optional[int] = Query(
        None,
        description="Localnet e2e: override similarity score when POC_E2E_SUBMIT_OVERRIDE=1",
    ),
    e2e_original_creator: Optional[str] = Query(
        None,
        description="Localnet e2e: override original_creator address",
    ),
    e2e_derivative_target: Optional[int] = Query(
        None,
        description="Localnet e2e: override derivative_redirection_target (0=wallet, 1=escrow)",
    ),
    upload_to_stream: bool = False,
    _rate_limit: None = Depends(check_rate_limit),
):
    """
    Synchronous upload endpoint (legacy/compatibility)
    
    Upload and analyze media file for similarity detection using PostgreSQL + pgvector.
    
    Supports:
    - Images: JPEG, PNG, GIF, WebP
    - Audio: MP3, WAV, FLAC, OGG  
    - Video: MP4, AVI, MOV, WebM
    
    For better UX with progress tracking, use /upload/stream instead
    
    If post_id is provided and MySocial integration is enabled, submits PoC result to blockchain.
    
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
        
        # Check if post was already analyzed (one media per post rule)
        if post_id and myso_client:
            try:
                post_check = myso_client.check_post_already_analyzed(post_id, force_reanalyze=force_reanalyze)
                if post_check.get("already_analyzed"):
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail=f"Post {post_id} was already analyzed by PoC. Only one media per post allowed."
                    )
            except HTTPException:
                raise
            except Exception as e:
                logger.warning("Could not check post status", post_id=post_id, error=str(e))
                # Continue anyway
        
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
            status="processing",
            creator_address=creator_address,
        )

        video_analysis = None
        matches: List[MediaMatch] = []
        if media_type == "image":
            matches = await detect_image_similarity(temp_file_path, media_id)
        elif media_type == "audio":
            matches = await detect_audio_similarity(temp_file_path, media_id)
        elif media_type == "video":
            video_analysis = analyze_video_similarity(temp_file_path, media_id)
            matches = video_analysis.matches
        
        # Upload to storage (simplified filename: just media_id + extension)
        from pathlib import Path
        ext = Path(file.filename).suffix or ".bin"
        simplified_filename = f"{media_id}{ext}"
        
        with open(temp_file_path, "rb") as f:
            storage_uri = storage_client.upload(simplified_filename, f, media_type)
        
        # Upload to Cloudflare Stream if requested (opt-in)
        streaming_uri = None
        if upload_to_stream and media_type == "video":
            try:
                logger.info("Uploading to Cloudflare Stream (opt-in enabled)",
                           media_id=media_id)
                file_size = os.path.getsize(temp_file_path)
                with open(temp_file_path, "rb") as f:
                    streaming_uri = storage_client.upload_to_stream(
                        simplified_filename,
                        f,
                        media_type,
                        file_size,
                        metadata={
                            "media_id": media_id,
                            "original_filename": file.filename,
                            "post_id": post_id if post_id else ""
                        }
                    )
                if streaming_uri:
                    logger.info("Video uploaded to Stream successfully",
                               media_id=media_id,
                               streaming_uri=streaming_uri)
            except Exception as e:
                logger.warning("Stream upload failed, continuing with R2 only",
                              media_id=media_id,
                              error=str(e))
        
        # Update media file with storage URI and completion status
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        update_media_file_status(
            media_id=media_id,
            status="completed",
            storage_uri=storage_uri,
            streaming_uri=streaming_uri,
            processing_results={
                "processing_time_ms": processing_time,
                "matches_found": len(matches),
                "media_type": media_type
            }
        )
        
        # Record attribution automatically based on similarity analysis
        media_type_code = {"image": 1, "audio": 3, "video": 2}.get(media_type, 1)
        poc_cfg = OFFCHAIN_DEFAULT_POC_CONFIG
        if myso_client:
            try:
                poc_cfg = myso_client.get_poc_config()
            except Exception:
                pass

        summary = preview_poc_chain_summary_for_upload(
            media_cat=media_type,
            media_type_code=media_type_code,
            matches=matches,
            video_analysis=video_analysis,
            poc_config=poc_cfg,
            royalty_free=royalty_free,
        )
        video_attestation_model = None
        if media_type == "video" and video_analysis is not None:
            video_attestation_model = preview_video_track_attestation(video_analysis, poc_cfg)

        tx_hash = None
        poc_submission_snap = None

        if post_id and myso_client:
            try:
                e2e_override = None
                if (
                    os.getenv("POC_E2E_SUBMIT_OVERRIDE", "").strip().lower() in ("1", "true", "yes")
                    and e2e_score is not None
                ):
                    creator_override = e2e_original_creator
                    if creator_override and creator_override.lower() in ("none", "null"):
                        creator_override = None
                    elif creator_override and creator_override.startswith("some("):
                        creator_override = creator_override[5:-1]
                    e2e_override = {
                        "media_type": media_type_code,
                        "score": e2e_score,
                        "original_creator": creator_override,
                        "derivative_target": e2e_derivative_target if e2e_derivative_target is not None else 0,
                        "royalty_free": royalty_free,
                    }
                myso_result, video_attestation_model, poc_submission_snap, summary = attempt_proof_of_creativity_submission(
                    myso_client=myso_client,
                    post_id=post_id,
                    media_cat=media_type,
                    media_type_code=media_type_code,
                    matches=matches,
                    video_analysis=video_analysis,
                    spt_pool_id=spt_pool_id,
                    royalty_free=royalty_free,
                    e2e_override=e2e_override,
                )
                tx_hash = myso_result.get("tx_hash")
                if poc_require_tx_when_post_id_from_env() and not tx_hash:
                    raise RuntimeError("MySocial PoC RPC returned no transaction digest (tx_hash)")
                logger.info(
                    "PoC result submitted to MySocial",
                    post_id=post_id,
                    tx_hash=tx_hash,
                    attribution_type=attribution_type_from_chain_summary(summary),
                )
            except Exception as e:
                if poc_require_tx_when_post_id_from_env():
                    err_msg = str(e)
                    logger.warning(
                        "MySocial PoC submission failed; oracle upload still completed "
                        "(MYSO_POC_REQUIRE_TX_WHEN_POST_ID=true)",
                        post_id=post_id,
                        error=err_msg,
                    )
                    fail_attribution_type = attribution_type_from_chain_summary(summary)
                    proof_fail = {
                        "file_hash": file_hash,
                        "original_filename": file.filename,
                        "upload_timestamp": time.time(),
                        "processing_time_ms": processing_time,
                        "matches_found": len(matches),
                        "media_type": media_type,
                        "high_confidence_matches": len(
                            [m for m in matches if m.confidence_level == "high"]
                        ),
                        "match_types": list(set(m.match_type for m in matches)) if matches else [],
                        "max_similarity_score": max([m.similarity_score for m in matches])
                        if matches
                        else 0.0,
                        "max_similarity_score_u64": summary.highest_similarity_score_u64,
                        "effective_threshold_u64": summary.effective_threshold_u64,
                        "poc_submission": None,
                        "poc_chain_summary": summary.model_dump(),
                        "royalty_free_requested": royalty_free,
                        "video_attestation": video_attestation_model.model_dump()
                        if video_attestation_model
                        else None,
                        "post_id": post_id,
                        "spt_pool_id": spt_pool_id,
                        "poc_submission_error": err_msg,
                        "chain_submission_failed": True,
                    }
                    try:
                        insert_attribution_record(
                            media_id=media_id,
                            attribution_type=fail_attribution_type,
                            proof_data=proof_fail,
                            tx_hash=None,
                        )
                    except Exception as att_e:
                        logger.warning(
                            "Could not record attribution after chain failure",
                            media_id=media_id,
                            error=str(att_e),
                        )
                    try:
                        update_media_file_status(
                            media_id,
                            "completed",
                            storage_uri=storage_uri,
                            streaming_uri=streaming_uri,
                            processing_results={
                                "processing_time_ms": processing_time,
                                "matches_found": len(matches),
                                "media_type": media_type,
                                "poc_submission_error": err_msg,
                                "chain_submission_failed": True,
                                "max_similarity_score_u64": summary.highest_similarity_score_u64,
                                "effective_threshold_u64": summary.effective_threshold_u64,
                                "poc_chain_summary": summary.model_dump(),
                            },
                        )
                    except Exception:
                        pass
                    base_msg = build_upload_attribution_message(matches, summary)
                    chain_suffix = (
                        f" MySocial PoC submission failed (required when post_id present): {err_msg}"
                    )
                    return UploadResponse(
                        media_id=media_id,
                        filename=file.filename,
                        content_type=content_type,
                        file_size=file_size,
                        file_hash=file_hash,
                        storage_uri=storage_uri,
                        streaming_uri=streaming_uri,
                        matches=matches,
                        processing_status="completed",
                        message=base_msg + chain_suffix,
                        tx_hash=None,
                        video_attestation=video_attestation_model,
                        poc_chain_summary=summary,
                        poc_submission_error=err_msg,
                    )
                logger.error(
                    "Failed to submit to MySocial blockchain",
                    post_id=post_id,
                    error=str(e),
                )
                summary = preview_poc_chain_summary_for_upload(
                    media_cat=media_type,
                    media_type_code=media_type_code,
                    matches=matches,
                    video_analysis=video_analysis,
                    poc_config=poc_cfg,
                    royalty_free=royalty_free,
                )
        attribution_type = attribution_type_from_chain_summary(summary)
        message = build_upload_attribution_message(matches, summary)

        proof_data = {
            "file_hash": file_hash,
            "original_filename": file.filename,
            "upload_timestamp": time.time(),
            "processing_time_ms": processing_time,
            "matches_found": len(matches),
            "media_type": media_type,
            "high_confidence_matches": len([m for m in matches if m.confidence_level == "high"]),
            "match_types": list(set(m.match_type for m in matches)) if matches else [],
            "max_similarity_score": max([m.similarity_score for m in matches]) if matches else 0.0,
            "max_similarity_score_u64": summary.highest_similarity_score_u64,
            "effective_threshold_u64": summary.effective_threshold_u64,
            "poc_submission": poc_submission_snap,
            "poc_chain_summary": summary.model_dump(),
            "royalty_free_requested": royalty_free,
            "video_attestation": video_attestation_model.model_dump()
            if video_attestation_model
            else None,
            "post_id": post_id,
            "spt_pool_id": spt_pool_id,
        }

        insert_attribution_record(
            media_id=media_id,
            attribution_type=attribution_type,
            proof_data=proof_data,
            tx_hash=tx_hash,
        )

        logger.info(
            "Attribution recorded",
            media_id=media_id,
            attribution_type=attribution_type,
            matches_found=len(matches),
            tx_hash=tx_hash,
        )

        return UploadResponse(
            media_id=media_id,
            filename=file.filename,
            content_type=content_type,
            file_size=file_size,
            file_hash=file_hash,
            storage_uri=storage_uri,
            streaming_uri=streaming_uri,
            matches=matches,
            processing_status="completed",
            message=message,
            tx_hash=tx_hash,
            video_attestation=video_attestation_model,
            poc_chain_summary=summary,
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
                similarity_score_percent=similarity_float_to_u64_percent(float(match["similarity_score"])),
                match_type=match["match_type"],
                confidence_level=match["confidence_level"],
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

@app.get("/upload/{upload_id}/progress")
async def get_upload_progress(upload_id: str):
    """
    Get current upload progress status (HTTP polling alternative to WebSocket)
    
    Returns current progress state
    """
    if not progress_tracker:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Progress tracking not available (Redis not configured)"
        )
    
    progress_data = progress_tracker.get_progress(upload_id)
    
    if not progress_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Upload {upload_id} not found or expired"
        )
    
    return progress_data

@app.websocket("/ws/upload/{upload_id}")
async def websocket_upload_progress(websocket: WebSocket, upload_id: str):
    """
    WebSocket endpoint for real-time upload progress streaming
    
    Connect to: ws://host:port/ws/upload/{upload_id}
    
    No authentication required
    Unlimited connections
    Streams progress until upload complete or 10 minute timeout
    """
    from app.api.websocket import stream_upload_progress
    
    await stream_upload_progress(websocket, upload_id, progress_tracker)

@app.get("/stats", response_model=dict)
async def get_system_stats():
    """Get comprehensive system statistics."""
    try:
        db_stats = get_database_stats()
        storage_health = storage_client.health_check() if storage_client else {}
        redis_health = redis_cache.health_check() if redis_cache else {"available": False}
        
        return {
            "database": db_stats,
            "storage": storage_health,
            "redis": redis_health,
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
    # Railway sets PORT, fallback to API_PORT or 8080
    port = int(os.getenv("PORT") or os.getenv("API_PORT") or "8080")
    
    uvicorn.run(
        "app.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=port,
        reload=os.getenv("DEBUG", "false").lower() == "true",
        log_config=None,  # We handle logging with structlog
    ) 