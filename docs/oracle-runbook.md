# Proof of Creativity Oracle — Operations Runbook

## Overview

The oracle listens for on-chain `PostCreatedEvent` entries (via gRPC sync), downloads `media_urls`, runs similarity analysis, provisions off-network username beneficiary vaults when required, and submits `analyze_and_update_post` as the configured oracle signer.

## Services

| Process | Command | Role |
|---------|---------|------|
| API | `python scripts/run_api.py` | REST `/oracle/*` + WebSocket `/ws` |
| gRPC sync | `python scripts/run_grpc_sync.py` | Ingest posts, enqueue jobs |
| Oracle worker | `python scripts/run_oracle_worker.py` | Analyze media, submit txs |

Docker Compose runs all services against local Postgres:

```bash
# 1) Ensure Docker Desktop is running (must respond quickly):
docker info

# 2) Infra only — starts in seconds, no image build:
docker compose up postgres redis

# 3) Full stack — first build downloads torch/CLIP (10–20 min):
docker compose --profile app up --build
# or: ./scripts/docker_up.sh
```

If `docker compose up` prints nothing for a long time, the usual causes are:
- Docker Desktop not running (`docker info` hangs)
- Missing `.dockerignore` uploading `.venv` (fixed in repo)
- First `--build` downloading large ML wheels (wait for pip output)

## Network selection

Set `MYSO_NETWORK=localnet|testnet|mainnet` or `MYSO_NETWORKS=localnet,testnet` for multi-network workers.

Object IDs and RPC/gRPC URLs live in `config/networks/{network}.yaml`.

On startup, each oracle process calls **session refresh** (when `MYSO_REFRESH_SESSION_OBJECTS=true`):

1. Query GraphQL at `GRAPHQL_URL` (default `http://127.0.0.1:9125/graphql`) for shared PoC objects:
   - `PoCConfig`, `PoCRegistry`, `PoCVaultDirectory`, `TokenRegistry`, etc.
2. Resolve `MYSO_POC_PACKAGE_ID` from the config object's on-chain Move type via RPC.
3. Fall back to `.env` / `config/networks/*.yaml` when GraphQL is unreachable.

Docker Compose sets `GRAPHQL_URL=http://host.docker.internal:9125/graphql` so containers reach the host indexer.

## Localnet gRPC sync

`config/networks/localnet.yaml` sets **`grpc_sync.mock_mode: false`** for live localnet with `myso start --with-poc` (checkpoint sync against the host fullnode at `http://host.docker.internal:9000`).

To replay fixtures offline instead of ingesting live `PostCreatedEvent` entries, explicitly set `grpc_sync.mock_mode: true` and point `fixture_path` at `data/fixtures/post_created_events.json`. Mock replay is **not** the default for localnet.

**Recommended E2E validation** after starting the stack:

```bash
# From myso-core (sibling proof-of-creativity repo required)
ASSUME_YES=1 ./scripts/poc-e2e-runnable.sh --run-all
```

This script syncs the on-chain oracle address, creates a PoC+SPT post, waits for grpc-sync + worker attestation, asserts Move events via `myso client tx-block`, and runs discovery embed + mock username claim legs.

## Production gRPC sync (default: checkpoint_v2)

`grpc-sync` uses standard v2 gRPC — no `authenticated_events_indexing` required:

1. **Catch-up:** `LedgerService.GetCheckpoint` for each sequence with transaction events
2. **Live tail:** `SubscriptionService.SubscribeCheckpoints` (falls back to polling if unavailable)
3. **Lag metrics:** `LedgerService.GetServiceInfo.checkpoint_height`

| Setting | Purpose |
|---------|---------|
| `grpc_sync.mock_mode` | `true` for fixture replay; `false` for live chain |
| `grpc_sync.sync_mode` | `checkpoint_v2` (default), `authenticated_events` (opt-in), or mock via `mock_mode` |
| `grpc_sync.event_stream_id` | Package filter; defaults to `MYSO_POC_PACKAGE_ID` |
| `grpc_sync.poll_interval_seconds` | Idle sleep when caught up (polling mode) |
| `grpc_sync.checkpoint_catchup_batch_size` | Checkpoints fetched per catch-up batch |

**Localnet live sync:** set `mock_mode: false` and `sync_mode: checkpoint_v2` in `config/networks/localnet.yaml`. Docker uses `grpc_url: http://host.docker.internal:9000`.

**Optional authenticated events mode:** set `sync_mode: authenticated_events` only when the fullnode has `authenticated_events_indexing = true`.

**Checkpointing:** Stored per `(network, stream_id)` in `grpc_sync_checkpoints`.

**Sync status:** `GET /oracle/sync/status?network=localnet` returns `sync_mode`, `chain_tip`, `lag_checkpoints`, and `checkpoint`.

## Mainnet writes

Mainnet profile sets `chain_writes_enabled: false`. Enable production writes only with:

```bash
MAINNET_WRITES_ENABLED=true
```

## WebSocket subscription

```javascript
const ws = new WebSocket("ws://localhost:8000/ws?network=localnet&topics=sync,jobs");
ws.onmessage = (e) => console.log(JSON.parse(e.data));
ws.send(JSON.stringify({
  action: "subscribe",
  network: "localnet",
  topics: ["post:0xmock_post_001", "sync"]
}));
```

Event types include: `post.discovered`, `post.analysis.progress`, `post.attestation.submitted`, `post.attestation.confirmed`, `sync.checkpoint`.

## REST endpoints

- `GET /oracle/networks`
- `GET /oracle/sync/status?network=localnet`
- `GET /oracle/posts/{post_id}?network=localnet`
- `GET /oracle/posts/{post_id}/proof`
- `POST /oracle/beneficiaries/{identity_hash}/claim?claimant_address=0x...`

## Database

Oracle tables are added via Alembic revision `c7d8e9f0a1b2`. Run migrations on startup (FastAPI lifespan) or:

```bash
pip install -r requirements-migrate.txt
alembic upgrade head
```

For full app setup use **Python 3.10–3.14**. Chain signing deps install via:

```bash
pip install -r requirements-blockchain.txt
```

Ed25519 oracle keys (default) work on all supported Python versions without `bip-utils`.  
secp256k1/secp256r1 **mnemonic** paths optionally need `pip install bip-utils` (Python 3.10–3.13 only); use `MYSO_ORACLE_PRIVATE_KEY` on Python 3.14+.

## Testing

```bash
pytest tests/unit/ -v
```
