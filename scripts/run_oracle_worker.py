#!/usr/bin/env python3
"""Run oracle analysis + chain submission worker."""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv

load_dotenv()


async def _main() -> None:
    from app.network_config import bootstrap_active_network_sessions

    bootstrap_active_network_sessions()
    from app.network_config import active_networks
    from app.workers.oracle_worker import run_oracle_workers

    await run_oracle_workers(active_networks())


if __name__ == "__main__":
    asyncio.run(_main())
