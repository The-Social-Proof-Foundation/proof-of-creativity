#!/usr/bin/env python3
"""Run blockchain sync worker (gRPC or social indexer GraphQL)."""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()


def _sync_mode() -> str:
    return os.getenv("POC_SYNC_MODE", "grpc").strip().lower()


async def _main() -> None:
    from app.network_config import active_networks, bootstrap_active_network_sessions

    bootstrap_active_network_sessions()
    networks = active_networks()
    mode = _sync_mode()

    if mode == "indexer":
        from app.chain.indexer_sync import run_indexer_sync_for_network

        await asyncio.gather(*(run_indexer_sync_for_network(n) for n in networks))
        return

    from app.workers.grpc_sync_worker import run_grpc_sync_workers

    await run_grpc_sync_workers(networks)


if __name__ == "__main__":
    asyncio.run(_main())
