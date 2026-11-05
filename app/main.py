import os
import structlog
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from dotenv import load_dotenv
load_dotenv()

from app.services.poc_utils import (
    mys_integration_enabled_from_env,
    mysocial_readiness_payload,
    poc_fail_lifespan_on_mys_misconfig_from_env,
)
from app.core.storage import StorageClient
from app.core.database import (
    get_database_stats,
    check_database_connection,
)
from app.models.similarity import (
    ErrorResponse,
    HealthResponse,
)

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

storage_client = None
redis_cache = None
myso_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    global storage_client, redis_cache, myso_client

    logger.info(
        "Starting Proof of Creativity API",
        main_module_path=os.path.abspath(__file__),
    )
    try:
        logger.info("Running database migrations...")
        from app.core.migrations import run_migrations
        migrations_ok = run_migrations()

        if migrations_ok:
            logger.info("✅ Database migrations completed")
        else:
            logger.warning("⚠️  Database migrations failed - continuing anyway")

        from app.core.redis_client import get_redis
        redis_cache = get_redis()
        if redis_cache.enabled:
            logger.info("✅ Redis cache initialized")
        else:
            logger.warning("⚠️  Redis cache disabled - performance will be slower")

        from app.core.database import initialize_connection_pool, create_vector_index_if_not_exists
        initialize_connection_pool()
        logger.info("Database connection pool initialized")

        try:
            create_vector_index_if_not_exists()
            logger.info("Vector indexes ensured")
        except Exception as e:
            logger.warning("Could not create vector indexes", error=str(e))

        from app.network_config import (
            active_networks,
            bootstrap_active_network_sessions,
            load_network_profile,
            validate_registry_objects,
        )
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

        storage_client = StorageClient()
        from app import config
        logger.info(
            "Storage client initialized",
            storage_backend="R2" if config.USE_CLOUDFLARE_R2 else "GCS" if config.USE_GCS else "Local",
        )

        if check_database_connection():
            logger.info("Database connection verified")
        else:
            logger.warning("Database connection check failed")

        from app.discovery.bootstrap import evaluate_bootstrap_status

        for network in active_networks():
            profile = load_network_profile(network)
            missing = validate_registry_objects(profile)
            if missing:
                logger.warning("Registry objects missing", network=network, missing=missing)

        discovery_status = evaluate_bootstrap_status()
        if discovery_status.ready:
            logger.info("Discovery bootstrap ready", assets=discovery_status.asset_counts)
        elif discovery_status.issues:
            logger.warning("Discovery bootstrap issues", issues=discovery_status.issues)

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

    logger.info("Shutting down Proof of Creativity API")


app = FastAPI(
    title="Proof of Creativity API",
    description=(
        "Oracle API for media fingerprinting, on-chain MediaAsset resolution, "
        "and reads for posts, assets, beneficiaries, reviews, and disputes."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    openapi_tags=[
        {
            "name": "oracle",
            "description": "MediaAsset resolution, posts, beneficiaries, reviews, and disputes.",
        },
        {
            "name": "health",
            "description": "Liveness, readiness, and process stats.",
        },
    ],
    responses={
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    }
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.api.rest.routes import router as oracle_rest_router
from app.api.ws.routes import router as oracle_ws_router

app.include_router(oracle_rest_router)
app.include_router(oracle_ws_router)


@app.get("/", response_model=dict, tags=["health"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Proof of Creativity API",
        "version": "1.0.0",
        "description": (
            "Chain-wide oracle: fingerprinting, similarity, and MediaAsset "
            "resolution for every social platform (DripDrop is one storage adapter)."
        ),
        "docs_url": "/docs",
        "health_url": "/health",
        "readyz_url": "/readyz",
        "oracle_url": "/oracle",
        "database": "PostgreSQL with pgvector",
        "publish": (
            "PostCreatedEvent (commitments in metadata_json) → "
            "oracle resolve_post_media / finalize_media_asset → analyze_post_composition"
        ),
    }


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint with comprehensive system status."""
    try:
        db_healthy = check_database_connection()
        db_stats = get_database_stats() if db_healthy else {}

        storage_health = storage_client.health_check() if storage_client else {"error": "not_initialized"}

        redis_health = redis_cache.health_check() if redis_cache else {"available": False}

        components = {
            "database": "healthy" if db_healthy else "unhealthy",
            "storage": "healthy" if storage_health.get("r2", {}).get("available") or storage_health.get("gcs", {}).get("available") else "unhealthy",
            "redis": "healthy" if redis_health.get("available") else "unavailable",
            "embedding_model": "healthy",
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


@app.get("/readyz", tags=["health"])
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


@app.get("/stats", response_model=dict, tags=["health"])
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


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "http_error", "message": exc.detail},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    if isinstance(exc, HTTPException):
        raise exc
    logger.error("Unhandled exception",
                url=str(request.url), method=request.method, error=str(exc), exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "internal_server_error", "message": "An unexpected error occurred"}
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT") or os.getenv("API_PORT") or "8080")

    uvicorn.run(
        "app.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=port,
        reload=os.getenv("DEBUG", "false").lower() == "true",
        log_config=None,
    )
