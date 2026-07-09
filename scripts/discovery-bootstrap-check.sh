#!/usr/bin/env bash
# Validate discovery bootstrap prerequisites for PoC + discovery-service.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

API_URL="${POC_API_URL:-http://127.0.0.1:8000}"
DISCOVERY_URL="${DISCOVERY_SERVICE_URL:-http://127.0.0.1:8096}"
SECRET="${DISCOVERY_EMBED_SECRET:-}"

echo "== Discovery bootstrap check =="

if [[ -z "$SECRET" ]]; then
  echo "WARN: DISCOVERY_EMBED_SECRET not set"
else
  echo "DISCOVERY_EMBED_SECRET: set"
fi

curl -sf "$API_URL/health" >/dev/null && echo "PoC API: ok" || {
  echo "PoC API health failed"
  exit 1
}

if [[ -n "$SECRET" ]]; then
  curl -sf -H "Authorization: Bearer $SECRET" "$API_URL/internal/discovery/status" | python3 -m json.tool
else
  echo "Skipping /internal/discovery/status (no embed secret)"
fi

curl -sf "$DISCOVERY_URL/health" >/dev/null && echo "discovery-service: ok" || {
  echo "WARN: discovery-service not reachable at $DISCOVERY_URL"
}

echo "Bootstrap check complete"
