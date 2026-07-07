#!/usr/bin/env python3
"""Run gRPC blockchain sync worker."""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()


async def _main() -> None:
    from app.network_config import bootstrap_active_network_sessions

    bootstrap_active_network_sessions()
    from app.network_config import active_networks
    from app.workers.grpc_sync_worker import run_grpc_sync_workers

    networks = active_networks()
    await run_grpc_sync_workers(networks)


if __name__ == "__main__":
    asyncio.run(_main())
