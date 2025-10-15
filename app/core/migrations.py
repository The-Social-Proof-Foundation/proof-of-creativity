"""
Auto-migration runner - similar to Diesel's embedded migrations
Runs database migrations automatically on application startup
"""
import os
from alembic import command
from alembic.config import Config
from pathlib import Path
import structlog

logger = structlog.get_logger()

def run_migrations():
    """
    Run all pending database migrations on startup.
    Similar to Diesel's embedded_migrations::run() in Rust.
    """
    try:
        # Get the alembic.ini path
        alembic_ini_path = Path(__file__).parent.parent.parent / "alembic.ini"
        
        if not alembic_ini_path.exists():
            logger.error("alembic.ini not found", path=str(alembic_ini_path))
            return False
        
        # Create Alembic config
        alembic_cfg = Config(str(alembic_ini_path))
        
        # Set the script location explicitly
        script_location = Path(__file__).parent.parent.parent / "alembic"
        alembic_cfg.set_main_option("script_location", str(script_location))
        
        logger.info("Running database migrations...")
        
        # Run migrations to HEAD (latest version)
        command.upgrade(alembic_cfg, "head")
        
        logger.info("✅ Database migrations completed successfully")
        return True
        
    except Exception as e:
        logger.error("❌ Database migration failed", error=str(e))
        # Don't fail the application - just log the error
        # This allows the app to start even if migrations fail
        return False

def check_migration_status():
    """Check if database is up to date with migrations"""
    try:
        alembic_ini_path = Path(__file__).parent.parent.parent / "alembic.ini"
        alembic_cfg = Config(str(alembic_ini_path))
        script_location = Path(__file__).parent.parent.parent / "alembic"
        alembic_cfg.set_main_option("script_location", str(script_location))
        
        # Get current revision
        from alembic.script import ScriptDirectory
        from alembic.runtime.migration import MigrationContext
        from sqlalchemy import create_engine
        
        # Get database URL
        db_url = os.getenv("DATABASE_URL") or os.getenv("DB_DSN")
        if not db_url:
            logger.warning("No database connection configured")
            return False
        
        # Fix Railway/Heroku format (postgres:// → postgresql://)
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        
        engine = create_engine(db_url)
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            current_rev = context.get_current_revision()
        
        script = ScriptDirectory.from_config(alembic_cfg)
        head_rev = script.get_current_head()
        
        is_up_to_date = current_rev == head_rev
        
        logger.info("Database migration status",
                   current_revision=current_rev or "None",
                   latest_revision=head_rev,
                   up_to_date=is_up_to_date)
        
        return is_up_to_date
        
    except Exception as e:
        logger.warning("Could not check migration status", error=str(e))
        return False

