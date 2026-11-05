"""
Alembic environment configuration - similar to Diesel migrations in Rust
Automatically generates migrations from SQLAlchemy models
"""
import importlib.util
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool
from alembic import context

# Add the parent directory to the path to import our models
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Load environment variables (.env optional — only python-dotenv required for migrate extras)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

# Import SQLAlchemy Base without pulling app.models (avoids pydantic/torch at migrate time)
_root = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "app.models.db_models",
    _root / "app" / "models" / "db_models.py",
)
if _spec is None or _spec.loader is None:
    raise ImportError("Could not load app/models/db_models.py for Alembic")
_db_models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_db_models)
Base = _db_models.Base

# this is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set target metadata from our models (like Diesel schema)
target_metadata = Base.metadata

# Get database URL from environment (Railway uses DATABASE_URL)
def get_url():
    url = os.getenv("DATABASE_URL") or os.getenv("TIMESCALE_DB_DSN") or os.getenv("DB_DSN") or \
          "postgresql://user:password@localhost:5432/proof_of_creativity"
    
    # Fix Railway/Heroku DATABASE_URL format (postgres:// → postgresql://)
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    
    return url

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.
    
    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,  # Detect column type changes
        compare_server_default=True,  # Detect default value changes
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    # Override the sqlalchemy.url with our environment variable
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_url()
    
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,  # Detect column type changes
            compare_server_default=True,  # Detect default value changes
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
