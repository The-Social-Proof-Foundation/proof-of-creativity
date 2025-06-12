#!/usr/bin/env python3
"""
Database initialization script for Proof of Creativity
Sets up Timescale database with vector extensions and hypertables
"""

import os
import sys
import psycopg2
from pathlib import Path

# Add the parent directory to Python path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import structlog

# Configure basic logging for the script
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Try to get database DSN from environment or use default
TIMESCALE_DB_DSN = os.getenv("TIMESCALE_DB_DSN", "postgresql://user:password@localhost:5432/proof_of_creativity")

def run_sql_file(cursor, filepath):
    """Execute SQL commands from a file."""
    try:
        with open(filepath, 'r') as f:
            sql_content = f.read()
        
        # Execute the entire file content at once to handle functions properly
        logger.info(f"Executing SQL file: {filepath}")
        cursor.execute(sql_content)
        
        logger.info(f"Successfully executed SQL file: {filepath}")
        
    except Exception as e:
        logger.error(f"Error executing SQL file {filepath}: {str(e)}")
        raise

def check_timescale_version(cursor):
    """Check if TimescaleDB is properly installed."""
    try:
        cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';")
        result = cursor.fetchone()
        if result:
            logger.info(f"TimescaleDB version: {result[0]}")
            return True
        else:
            logger.warning("TimescaleDB extension not found")
            return False
    except Exception as e:
        logger.error(f"Error checking TimescaleDB version: {str(e)}")
        return False

def check_vector_extension(cursor):
    """Check if pgvector is properly installed."""
    try:
        cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
        result = cursor.fetchone()
        if result:
            logger.info(f"pgvector version: {result[0]}")
            return True
        else:
            logger.warning("pgvector extension not found")
            return False
    except Exception as e:
        logger.error(f"Error checking pgvector version: {str(e)}")
        return False

def create_hypertables(cursor):
    """Create hypertables if they don't exist."""
    hypertables = [
        ("media_files", "created_at"),
        ("media_embeddings", "uploaded_at"),
        ("audio_fingerprints", "created_at"),
        ("similarity_matches", "created_at"),
        ("attribution_records", "created_at")
    ]
    
    for table_name, time_column in hypertables:
        try:
            # Check if hypertable already exists
            cursor.execute("""
                SELECT * FROM timescaledb_information.hypertables 
                WHERE hypertable_name = %s
            """, (table_name,))
            
            if cursor.fetchone():
                logger.info(f"Hypertable {table_name} already exists")
            else:
                # Create hypertable
                cursor.execute(f"""
                    SELECT create_hypertable('{table_name}', '{time_column}', 
                        chunk_time_interval => INTERVAL '1 day',
                        if_not_exists => TRUE
                    );
                """)
                logger.info(f"Created hypertable: {table_name}")
                
        except Exception as e:
            logger.error(f"Error creating hypertable {table_name}: {str(e)}")
            # Continue with other tables
            continue

def create_vector_indexes(cursor):
    """Create vector indexes if they don't exist."""
    try:
        # Check if HNSW index exists
        cursor.execute("""
            SELECT indexname FROM pg_indexes 
            WHERE indexname = 'idx_media_embeddings_vector_hnsw'
        """)
        
        if cursor.fetchone():
            logger.info("HNSW vector index already exists")
        else:
            # Create HNSW index
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_media_embeddings_vector_hnsw 
                ON media_embeddings USING hnsw (embedding vector_cosine_ops) 
                WITH (m = 16, ef_construction = 64);
            """)
            logger.info("Created HNSW vector index")
            
    except Exception as e:
        logger.error(f"Error creating vector indexes: {str(e)}")
        raise

def check_environment():
    """Check if environment is properly configured."""
    logger.info("Checking environment configuration...")
    
    # Check if .env file exists
    env_file = Path(__file__).parent.parent / ".env"
    if not env_file.exists():
        logger.warning(".env file not found. Using default configuration.")
        logger.info("💡 Create .env file from env.example for custom configuration")
    
    # Check database DSN
    if TIMESCALE_DB_DSN == "postgresql://user:password@localhost:5432/proof_of_creativity":
        logger.warning("Using default database DSN. Update TIMESCALE_DB_DSN environment variable.")
        logger.info("💡 Set TIMESCALE_DB_DSN in your .env file")
    
    return True

def test_database_connection():
    """Test database connection without requiring specific extensions."""
    try:
        logger.info(f"Testing connection to database...")
        conn = psycopg2.connect(TIMESCALE_DB_DSN)
        cursor = conn.cursor()
        
        # Test basic connection
        cursor.execute("SELECT version();")
        pg_version = cursor.fetchone()[0]
        logger.info(f"✅ PostgreSQL connected: {pg_version}")
        
        conn.close()
        return True
        
    except psycopg2.OperationalError as e:
        logger.error(f"❌ Database connection failed: {str(e)}")
        logger.info("💡 Please check your database configuration:")
        logger.info("   - Is PostgreSQL running?")
        logger.info("   - Is the database created?")
        logger.info("   - Are the connection details correct?")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected database error: {str(e)}")
        return False

def initialize_database():
    """Main database initialization function."""
    logger.info("🧠 Proof of Creativity - Database Initialization")
    logger.info("=" * 50)
    
    # Check environment
    if not check_environment():
        return False
    
    # Get schema file path (try clean schema first)
    schema_file = Path(__file__).parent.parent / "schema_clean.sql"
    if not schema_file.exists():
        # Fallback to original schema
        schema_file = Path(__file__).parent.parent / "schema.sql"
        if not schema_file.exists():
            logger.error(f"Schema file not found: {schema_file}")
            return False
    
    # Test basic connection first
    if not test_database_connection():
        return False
    
    try:
        # Connect to database
        conn = psycopg2.connect(TIMESCALE_DB_DSN)
        conn.autocommit = True
        cursor = conn.cursor()
        
        logger.info("🔗 Connected to database successfully")
        
        # Check PostgreSQL version
        cursor.execute("SELECT version();")
        pg_version = cursor.fetchone()[0]
        logger.info(f"📊 PostgreSQL version: {pg_version}")
        
        # Check TimescaleDB installation
        timescale_ok = check_timescale_version(cursor)
        vector_ok = check_vector_extension(cursor)
        
        if not timescale_ok:
            logger.warning("⚠️ TimescaleDB not found. Schema will use regular PostgreSQL.")
            logger.info("💡 For optimal performance, consider using Timescale Cloud: https://console.cloud.timescale.com/")
            
        if not vector_ok:
            logger.warning("⚠️ pgvector not found. Vector similarity search will not work.")
            logger.info("💡 Install pgvector: https://github.com/pgvector/pgvector")
        
        # Execute schema file
        logger.info("📋 Executing schema.sql...")
        run_sql_file(cursor, schema_file)
        
        # Try to verify hypertables if TimescaleDB is available
        if timescale_ok:
            try:
                cursor.execute("SELECT * FROM timescaledb_information.hypertables;")
                hypertables = cursor.fetchall()
                logger.info(f"📈 Hypertables created: {len(hypertables)}")
                for ht in hypertables:
                    logger.info(f"   - {ht[1]} (schema: {ht[0]})")
            except Exception as e:
                logger.warning(f"Could not verify hypertables: {str(e)}")
        
        # Try to verify vector indexes if pgvector is available
        if vector_ok:
            try:
                cursor.execute("""
                    SELECT indexname, tablename FROM pg_indexes 
                    WHERE indexname LIKE '%vector%'
                """)
                vector_indexes = cursor.fetchall()
                logger.info(f"🔍 Vector indexes: {len(vector_indexes)}")
                for idx in vector_indexes:
                    logger.info(f"   - {idx[0]} on {idx[1]}")
            except Exception as e:
                logger.warning(f"Could not verify vector indexes: {str(e)}")
        
        # Verify basic tables
        cursor.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        """)
        tables = cursor.fetchall()
        logger.info(f"📋 Tables created: {len(tables)}")
        for table in tables:
            logger.info(f"   - {table[0]}")
        
        logger.info("✅ Database initialization completed successfully!")
        logger.info("🚀 You can now start the development server with: python scripts/run_dev.py")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {str(e)}")
        logger.info("💡 Common solutions:")
        logger.info("   - Check your database connection string")
        logger.info("   - Ensure the database exists")
        logger.info("   - Verify permissions")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize Proof of Creativity database")
    parser.add_argument("--force", action="store_true", 
                       help="Force recreation of database objects")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    success = initialize_database()
    sys.exit(0 if success else 1) 