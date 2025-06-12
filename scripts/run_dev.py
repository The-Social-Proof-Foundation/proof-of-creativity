#!/usr/bin/env python3
"""
Development server runner for Proof of Creativity API
Includes auto-reload, logging, and environment checking
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

def check_environment():
    """Check if all required environment variables are set."""
    required_vars = [
        "TIMESCALE_DB_DSN",
    ]
    
    optional_vars = [
        "API_HOST",
        "API_PORT", 
        "DEBUG",
        "USE_GCS",
        "USE_WALRUS",
        "GCS_BUCKET_NAME",
        "WALRUS_ENDPOINT"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file or environment configuration.")
        return False
    
    print("✅ Required environment variables found")
    
    # Show optional vars status
    print("\n📋 Optional configurations:")
    for var in optional_vars:
        value = os.getenv(var, "Not set")
        if var in ["TIMESCALE_DB_DSN"] and value != "Not set":
            # Don't show full database URL for security
            value = f"{value[:20]}..." if len(value) > 20 else value
        print(f"  {var}: {value}")
    
    return True

def check_dependencies():
    """Check if all required dependencies are available."""
    required_modules = [
        "fastapi",
        "uvicorn", 
        "psycopg2",
        "torch",
        "transformers",
        "PIL",  # Pillow imports as PIL
        "numpy",
        "librosa",
        "structlog"
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"❌ Missing required Python modules: {', '.join(missing_modules)}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    print("✅ All required dependencies found")
    return True

def main():
    """Main entry point for development server."""
    print("🧠 Proof of Creativity - Development Server")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Test database connection
    try:
        from app.core.database import check_database_connection
        if check_database_connection():
            print("✅ Database connection successful")
        else:
            print("❌ Database connection failed")
            print("Please check your database configuration")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Database connection error: {str(e)}")
        sys.exit(1)
    
    # Get configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    debug = os.getenv("DEBUG", "true").lower() == "true"
    
    print(f"\n🚀 Starting development server...")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Debug: {debug}")
    print(f"   Docs: http://{host}:{port}/docs")
    print(f"   API: http://{host}:{port}")
    print("\n⏹️  Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start the server
    try:
        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            reload=debug,
            log_level="debug" if debug else "info",
            access_log=True,
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 