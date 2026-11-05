#!/usr/bin/env python3
"""Run FastAPI API server (REST + WebSocket)."""

from __future__ import annotations

import os

import uvicorn
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    port = int(os.getenv("PORT") or os.getenv("API_PORT") or "8000")
    uvicorn.run(
        "app.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=port,
        reload=os.getenv("DEBUG", "false").lower() == "true",
        log_config=None,
    )


if __name__ == "__main__":
    main()
