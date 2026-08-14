# PoC Oracle Runbook (MediaAsset-centric)

## Overview

The oracle ingests on-chain social events via gRPC checkpoint sync and runs two job types:

| Job | Trigger | Chain entry |
|-----|---------|-------------|
| `resolve_media_asset` | `MediaResolutionRequestedEvent` | `media_asset::finalize_media_asset` |
| `analyze_composition` | `PostCreatedEvent` with `media_asset_ids` | `proof_of_creativity::analyze_post_composition` |
| `poc_gov_finalize_rights` | PoC governance proposal past `votingEndTime` (status=2) | `proof_of_creativity::finalize_media_asset_rights_governance_proposal` |
| `poc_gov_implement_rights` | Approved rights dispute (status=3) + stored claims bundle | `proof_of_creativity::finalize_media_asset_rights_via_dao` |

Legacy `analyze_post` jobs (post-centric, `media_urls` only) are deprecated.

## Event sync

Configured in network profile (`grpc_sync` section). Live localnet uses checkpoint streaming; mock mode replays fixtures.

Indexed event types:

- `MediaResolutionRequestedEvent`
- `MediaAssetResolvedEvent`, `FingerprintLinkedEvent`
- `MediaAssetUsedEvent`
- `PostCreatedEvent` (must decode `media_asset_ids`, `composition_status`, `monetization_status`)
- `PostCompositionAnalyzedEvent`

## Asset resolution pipeline

1. Client submits `submit_media_resolution`.
2. Worker loads job payload (`request_id`, fingerprint commitments, `media_type`).
3. Off-chain dedup via `fingerprint_observations` / vector store → optional `link_to_existing_id`.
4. Oracle submits `finalize_media_asset` with default or custom `claims` + `usage_grants`.
5. Upsert `chain_media_assets` with `rights_json` containing grants.

## Composition pipeline

1. `PostCreatedEvent` carries `media_asset_ids`.
2. Worker validates each asset exists and permits `SOCIAL_POST` usage class via `usage_grants`.
3. Optional per-URL similarity for derivative detection within composition.
4. Build `CompositionAnalysis` (version pins) + `RevenueManifest` (attributable pool splits).
5. Submit `analyze_post_composition` or `_sync_token_pool` when `spt_id` present.

## Prospective rights (V1)

When a referenced asset's `rights_version` or `economics_version` increments after composition analysis:

- `monetization_status → RESTRICTED`
- `composition_status` may stay `VERIFIED`
- No retroactive clawback of historical tips

## Media asset rights governance (DAO disputes)

Rights disputes use the PoC `GovernanceDAO` (separate fee: `media_asset_dispute_cost` on chain).

### Client / operator flow

1. `POST /oracle/disputes/media-asset-rights/prepare` — compute `claims_commitment` (SHA3-256 of BCS `ClaimsBundle`).
2. User submits on-chain `submit_media_asset_rights_dispute_proposal` with payment + commitment.
3. `POST /oracle/disputes/media-asset-rights/submit` — store full claims bundle in Postgres (`media_asset_rights_bundles`).
4. Worker polls indexer GraphQL for PoC registry proposals (`registryType=1`):
   - **Voting ended** (`status=2`, `votingEndTime < now`) → enqueue `poc_gov_finalize_rights`.
   - **Approved** (`status=3`) with bundle `status=pending` → enqueue `poc_gov_implement_rights`.
   - **Rejected / implemented** → update bundle status off-chain.

### REST endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/oracle/disputes/media-asset-rights/prepare` | Commitment preview |
| POST | `/oracle/disputes/media-asset-rights/submit` | Persist bundle after on-chain submit |
| GET | `/oracle/disputes/media-asset-rights/{proposal_id}/bundle` | Operator inspect stored bundle |

### Governance env

- `POC_GOVERNANCE_REGISTRY_ID` — PoC `GovernanceDAO` object id (`PoCConfig.dispute_governance_registry_id`)
- `POC_GOVERNANCE_POLL_INTERVAL_SECS` (default `30`)
- `POC_GOVERNANCE_IMPLEMENT_ENABLED` (default `true`)
- `POC_POST_REFRESH_AFTER_RIGHTS_IMPLEMENT` (default `true`) — enqueue `refresh_post_usage_decisions` for posts embedding the asset
- `MYSO_ECOSYSTEM_TREASURY_ID`, `GRAPHQL_URL`

Oracle signer must match `PoCConfig.oracle_address`.

## Required env

- `MYSO_POC_PACKAGE_ID`, `MYSO_POC_CONFIG_ID`, `MYSO_POC_REGISTRY_ID`, `MYSO_POC_VAULT_DIRECTORY_ID`
- Oracle signer matching on-chain `PoCConfig.oracle_address`
- `max_embedded_asset_redirect_bps` on `PoCConfig` (default 5000): caps each embedded source asset's manifest slice and license `compensation_bps`; oracle clamps manifests before submit
- `MYSO_INTEGRATION_ENABLED=true` for chain submission

## Verification

```bash
pytest tests/integration/test_media_asset_e2e_flow.py tests/unit/test_bcs_post_created.py -q
```

See also: [`media-asset-client-flow.md`](media-asset-client-flow.md)
