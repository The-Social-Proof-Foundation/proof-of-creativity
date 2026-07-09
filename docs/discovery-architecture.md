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
5. **PoC embed is creative media only** — `media_type` must be `image` / `audio` / `video` (or `1`/`2`/`3`). Factual RSS/JSON is SPoT’s lane; non-media requests get **400**, unreadable downloads get **422**.

## Two corpora

| Corpus | Owner | Config (myso-core) |
|--------|-------|--------------------|
| Creative media → PoC embed | discovery-service + PoC | `sources.media.localnet.yaml` |
| Factual text → SPoT settlement | spot-oracle `TrustedSource` | `sources.factual.localnet.yaml` / spot YAML |

## Data flow

```
DiscoverySource (creative media only for embed)
  → normalize (content_kind=media) → lifecycle FSM → discovery_jobs
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

SPoT uses shared YAML/`HttpFetchClient` from `myso-discovery-service-core` and its own `TrustedSource` adapters for **factual text/price/events**. It does **not** consume the PoC media embed corpus or `discovery_assets`.

## Related docs

- Oracle runbook: `docs/oracle-runbook.md`
- Discovery runbook: `docs/discovery-runbook.md`
- myso-core discovery ARCHITECTURE: `myso-core/crates/myso-discovery-service/docs/ARCHITECTURE.md`
