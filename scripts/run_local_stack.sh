#!/usr/bin/env bash
# Run the full PoC stack natively (no Docker): grpc-sync, oracle-worker,
# discovery-worker, and API. Requires local Postgres + myso localnet.
#
# Usage:
#   ./scripts/run_local_stack.sh
#   DISCOVERY_ENABLED=false ./scripts/run_local_stack.sh
#   POC_LOCAL_STACK_API_RELOAD=1 ./scripts/run_local_stack.sh   # hot-reload API
#
# Prerequisites:
#   - .venv activated or on PATH
#   - DATABASE_URL pointing at local Postgres (postgresql://postgres:postgres@localhost:5432/proof_of_creativity)
#   - myso localnet: RPC :9000, GraphQL :9125

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

# Native localnet defaults (Docker compose uses host.docker.internal instead).
export MYSOCIAL_RPC_URL="${MYSOCIAL_RPC_URL:-http://127.0.0.1:9000}"
export MYSO_GRPC_URL="${MYSO_GRPC_URL:-$MYSOCIAL_RPC_URL}"
export MYSO_GRPC_TLS="${MYSO_GRPC_TLS:-false}"
export GRAPHQL_URL="${GRAPHQL_URL:-http://127.0.0.1:9125/graphql}"

API_PORT="${API_PORT:-${PORT:-8001}}"
PIDS=()

cleanup() {
  echo ""
  echo "Stopping PoC local stack..."
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
  echo "Stopped."
  exit 0
}

trap cleanup INT TERM

prefix_logs() {
  local name=$1
  while IFS= read -r line || [[ -n "$line" ]]; do
    printf '[%s] %s\n' "$name" "$line"
  done
}

start_service() {
  local name=$1
  shift
  echo "Starting ${name}..."
  (
    exec "$@"
  ) > >(prefix_logs "$name") 2>&1 &
  PIDS+=("$!")
}

echo "PoC local stack — project: ${ROOT}"
echo "  DATABASE_URL=${DATABASE_URL:-<not set>}"
echo "  MYSOCIAL_RPC_URL=${MYSOCIAL_RPC_URL}"
echo "  GRAPHQL_URL=${GRAPHQL_URL}"
echo ""

if [[ -z "${DATABASE_URL:-}" && -z "${TIMESCALE_DB_DSN:-}" && -z "${DB_DSN:-}" ]]; then
  echo "ERROR: Set DATABASE_URL in .env (postgresql://postgres:postgres@localhost:5432/proof_of_creativity)" >&2
  exit 1
fi

if command -v alembic >/dev/null 2>&1; then
  echo "Applying database migrations (alembic upgrade heads)..."
  if ! alembic upgrade heads; then
    echo "WARNING: migrations failed — services will start anyway" >&2
  fi
  echo ""
fi

start_service grpc-sync python scripts/run_grpc_sync.py
sleep 1
start_service oracle-worker python scripts/run_oracle_worker.py
sleep 1

if [[ "${DISCOVERY_ENABLED:-true}" != "false" ]]; then
  start_service discovery-worker python scripts/run_discovery_worker.py
  sleep 1
fi

if [[ "${POC_LOCAL_STACK_API_RELOAD:-}" == "1" ]]; then
  start_service api env DEBUG=true python scripts/run_dev.py
else
  start_service api env DEBUG=false PORT="${API_PORT}" API_PORT="${API_PORT}" python scripts/run_api.py
fi

echo ""
echo "PoC local stack running — press Ctrl+C to stop all services"
echo "  API docs: http://127.0.0.1:${API_PORT}/docs"
echo "  Health:   http://127.0.0.1:${API_PORT}/health"
echo ""

wait
