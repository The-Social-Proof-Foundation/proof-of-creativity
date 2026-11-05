# Off-network creative media discovery (self-contained in PoC).

## Overview

Proof of Creativity owns the full off-network flow:

1. **Discovery scheduler** polls creative media YAML sources (`manual_curated` for localnet E2E).
2. **Discovery worker** enqueues and processes `embed_asset` jobs internally (no HTTP to external discovery service).
3. Embeddings land in pgvector with `corpus_scope=discovered` and `discovery_asset_id` pointing at `discovery_assets`.
4. **Oracle worker** searches platform + discovered corpora, provisions lazy vaults, submits chain transactions.
5. **Social indexer** indexes resulting on-chain PoC events (unchanged).

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `DISCOVERY_ENABLED` | `true` | Enable discovery scheduler |
| `DISCOVERY_EMBED_ENABLED` | `true` | Enqueue embed jobs for creative media |
| `DISCOVERY_SOURCES_CONFIG` | `config/discovery/sources.localnet.yaml` | Source registry |
| `DISCOVERY_POLL_INTERVAL_SECONDS` | `30` | Scheduler poll interval |
| `POC_USE_MANUAL_CURATED` | `1` (localnet) | Gate manual_curated adapter |
| `POC_ACTIVE_EMBEDDING_VERSION` | `clip-vit-b32-v1` | Active embedding version for similarity search |

## Docker services

```bash
docker compose --profile app up --build
```

Runs: `postgres`, `redis`, `api`, `grpc-sync`, `oracle-worker`, `discovery-worker`.

## Hard rules

1. PoC never calls external discovery services.
2. Raw external media is not persisted — temp fetch → embed → delete.
3. Lazy vault provisioning runs only from `oracle_worker` on off-network derivative matches.
4. On-chain Move modules and social indexer handlers are unchanged.
