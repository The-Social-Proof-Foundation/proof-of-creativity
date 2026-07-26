#!/usr/bin/env bash
# Verify PoC API readiness and discovery migration state.

set -euo pipefail

POC_URL="${POC_ORACLE_URL:-http://127.0.0.1:8000}"

echo ">>> PoC discovery bootstrap check"
echo "POC_URL=$POC_URL"

curl -sf "$POC_URL/health" >/dev/null && echo "poc-api: ok" || {
  echo "ERROR: PoC API not reachable at $POC_URL"
  exit 1
}

curl -sf "$POC_URL/oracle/health" >/dev/null && echo "oracle: ok" || echo "WARN: /oracle/health unavailable"

echo "Run alembic upgrade head if discovery_assets table is missing (migration f3a4b5c6d7e8)."
