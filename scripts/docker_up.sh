#!/usr/bin/env bash
# Start PoC oracle docker stack with visible progress.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Checking Docker daemon..."
if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker is not responding. Start Docker Desktop, wait until it is ready, then retry."
  exit 1
fi

PROFILE="${1:-app}"
export DOCKER_BUILDKIT=1
export COMPOSE_BAKE=false

if [[ "$PROFILE" == "infra" ]]; then
  echo "Starting postgres + redis only..."
  exec docker compose up postgres redis
fi

echo "Starting full stack (profile=app). First build may take 10–20 minutes (torch/CLIP)..."
exec docker compose --profile app up --build
