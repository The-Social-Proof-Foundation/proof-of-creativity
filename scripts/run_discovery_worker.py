#!/usr/bin/env python3
"""Run the off-network discovery worker (scheduler + embed jobs)."""

from __future__ import annotations

import asyncio
import sys

from app.workers.discovery_worker import run_discovery_worker


def main() -> int:
    try:
        asyncio.run(run_discovery_worker())
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"discovery worker failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
