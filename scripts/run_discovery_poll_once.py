#!/usr/bin/env python3
"""One-shot discovery scheduler poll (for E2E / ops)."""

from __future__ import annotations

import json
import sys

from dotenv import load_dotenv

load_dotenv()


def main() -> int:
    from app.discovery.scheduler import DiscoveryScheduler

    stats = DiscoveryScheduler().poll_once()
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
