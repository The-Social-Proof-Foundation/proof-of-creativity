"""gRPC sync worker entrypoint."""

from __future__ import annotations

import asyncio

import structlog

from app.chain.grpc_sync import run_sync_for_network

logger = structlog.get_logger()


async def run_grpc_sync_workers(networks: list[str]) -> None:
    logger.info("Starting gRPC sync workers", networks=networks)
    await asyncio.gather(*(run_sync_for_network(n) for n in networks))
