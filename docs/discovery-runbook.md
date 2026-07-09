# Proof of Creativity — Discovery Production Notes

## Services

| Service | Repo | Port | Command |
|---------|------|------|---------|
| PoC API + embed | proof-of-creativity | 8000 (or 8001) | `python scripts/run_api.py` / docker compose |
| Oracle worker | proof-of-creativity | — | `python scripts/run_oracle_worker.py` |
| gRPC sync worker | proof-of-creativity | — | `python scripts/run_grpc_sync.py` |
| Discovery platform | myso-core | 8096 | `./scripts/run-discovery-service.sh` |
| Identity verification | myso-identity-verification | 3000 | `./myso-identity-verification` |

## Enable discovery (media corpus for PoC)

1. Start discovery Postgres: `docker compose -f crates/myso-discovery-service/docker-compose.yml up discovery-postgres -d`
2. Set matching secrets in PoC `.env`: `DISCOVERY_EMBED_SECRET`, `DISCOVERY_SERVICE_URL=http://127.0.0.1:8096`
3. Run bootstrap check: `./scripts/discovery-bootstrap-check.sh`
4. Cold-start corpus (discovery-service): `DISCOVERY_BOOTSTRAP_ONLY=true DISCOVERY_EMBED_ENABLED=true ./myso-discovery-service`
5. Run media E2E from myso-core (curated images, not factual RSS):

```bash
DISCOVERY_EMBED_SECRET=<same-as-poc> \
DISCOVERY_EMBED_ENDPOINT=http://127.0.0.1:8000/internal/discovery/embed \
./scripts/discovery-poc-runnable.sh
```

**Integrated PoC + discovery + claim loop** (recommended after `myso start --with-poc`):

```bash
ASSUME_YES=1 ./scripts/poc-e2e-runnable.sh --run-all
```

Use `--skip-discovery` or `--skip-claim` for partial runs. Claim-only from the PoC repo: `./scripts/poc-claim-runnable.sh`.

PoC accepts **image/audio/video only**. Factual text discovery is for SPoT settlement, not this embed path.

## Status endpoints

- PoC: `GET /internal/discovery/status` (Bearer `DISCOVERY_EMBED_SECRET`)
- Discovery: `GET /health`

## Lazy vault rule

Vaults are provisioned only from `oracle_worker` when an off-network derivative match passes both `DISCOVERY_X_HANDLE_CONFIDENCE_THRESHOLD` and `DISCOVERY_WORK_CONFIDENCE_THRESHOLD`.

## Claim flow (prod)

1. `GET /oracle/beneficiaries/{identity_hash}/claim/status?claimant_address=0x...`
2. OAuth via myso-identity-verification: `POST /oauth/x/connect-for-poc-claim`
3. `POST /oracle/beneficiaries/{identity_hash}/claim` with session JWT

See `docs/poc-claim-evidence-v1.md`.

## Review queue

- `GET /oracle/reviews` — posts with `analysis_status=needs_review`
- `POST /oracle/reviews/{post_id}/resolve` — `{action: approve_escrow|reject|requeue}`

## Migrations

- PoC: Alembic revisions `d1e2f3a4b5c6` (corpus + provenance), `e2f3a4b5c6d7` (review columns)
- Discovery: SQLx migration in `myso-discovery-service-schema/migrations/`

## Re-embed

`python scripts/run_discovery_reembed_worker.py` — re-embed stale `embedding_version` rows.
