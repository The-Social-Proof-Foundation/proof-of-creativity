# Discovery & Hidden Provenance Architecture

Cross-repo map for the Proof of Creativity discovery platform and hidden provenance index.

## Repositories

| Component | Location | Role |
|-----------|----------|------|
| Discovery platform | `myso-core/crates/myso-discovery-service` | Source adapters, lifecycle FSM, scheduling, provenance index |
| ML + similarity | `proof-of-creativity` | CLIP/fingerprint embedding, pgvector, oracle worker, lazy vaults |
| Identity claims | `proof-of-creativity` + `myso-identity-verification` | X OAuth beneficiary claims |

## Hard rules

1. **Discovery never calls chain write APIs** — no vault provisioning during ingest.
2. **Lazy vaults** — `UsernameBeneficiaryService.ensure_provisioned` runs only from `oracle_worker` on off-network derivative matches.
3. **No raw external media persistence** — temp fetch → embed → delete; store embeddings, hashes, metadata, source URLs only.
4. **Hidden by default** — discovered creators and source URLs are internal; no public PoC badges for unverified creators.

## Data flow

```
DiscoverySource adapters (Rust)
  → normalize → lifecycle FSM → discovery_jobs queue
  → embed_client → POST /internal/discovery/embed (PoC)
  → pgvector corpus (corpus_scope=discovered, disc_* media_id)
  → oracle_worker similarity search (platform + discovered)
  → provenance_hit + lazy vault on infringement
```

## Identity hash (X username)

Move-compatible beneficiary identity uses canonical lowercase X handle bytes:

```python
from app.chain.move_address import canonical_registry_username, parse_identity_hash
handle = canonical_registry_username("CreatorName")
identity_hash = "0x" + parse_identity_hash(handle).hex()
```

Golden tests: `tests/unit/test_move_address.py`, `tests/unit/test_discovery_identity.py`.

## Embedding versioning

Active search uses `DISCOVERY_ACTIVE_EMBEDDING_VERSION` (default `clip-vit-b32-v1`). Re-embed jobs target new versions without deleting historical rows.

## Confidence model

- **work_confidence** — similarity / duplicate likelihood for the media itself.
- **creator_confidence** — attribution to a creator candidate (X handle anchor).

Vault provisioning requires both passing configured thresholds plus resolvable `identity_hash`.

## SPoT alignment

Discovery adapters mirror the SPoT oracle `TrustedSource` registry pattern (`DiscoverySource` trait). Factual adapters register disabled for future Social Proof of Truth consumption.

## Related docs

- Oracle runbook: `docs/oracle-runbook.md`
- SPoT oracle plan: `myso-core/.cursor/plans/spot_oracle_server_v1_17a97a27.plan.md`
