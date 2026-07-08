# Proof of Creativity — Discovery Production Notes

## Services

| Service | Repo | Port | Command |
|---------|------|------|---------|
| PoC API + embed | proof-of-creativity | 8000 | `python scripts/run_api.py` |
| Oracle worker | proof-of-creativity | — | `python scripts/run_oracle_worker.py` |
| Discovery platform | myso-core | 8096 | `./scripts/run-discovery-service.sh` |

## Enable discovery

1. Start discovery Postgres: `docker compose -f crates/myso-discovery-service/docker-compose.yml up discovery-postgres -d`
2. Set matching secrets in PoC `.env`: `DISCOVERY_EMBED_SECRET`, `DISCOVERY_SERVICE_URL=http://127.0.0.1:8096`
3. Run discovery service and PoC API stack.

## Lazy vault rule

Vaults are provisioned only from `oracle_worker` when an off-network derivative match passes `creator_confidence` and `DISCOVERY_X_HANDLE_CONFIDENCE_THRESHOLD`.

## Migrations

- PoC: Alembic revision `d1e2f3a4b5c6` adds corpus scoping + `provenance_hits`
- Discovery: SQLx migration in `myso-discovery-service-schema/migrations/`
