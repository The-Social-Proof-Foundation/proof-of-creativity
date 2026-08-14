# MediaAsset-Centric PoC Client Flow

This document describes the V1 end-to-end sequence for DripDrop and other clients after the MediaAsset architecture clean break.

## Protocol hierarchy

```text
MediaAsset.id          = canonical creative identity
Claim                  = asserted protocol statement (authorship, ownership, licensing authority, …)
RightsInterest[]       = resolved from verified ownership/control claims
UsageGrant             = license terms per product usage class
MediaAssetUsedEvent    = usage graph (post/profile/article containers)
CompositionAnalysis    = version-pinned multi-asset validation
RevenueManifest        = creator-attributable pool splits only
Post.monetization_status / composition_status = composition outcomes
```

Protocol and platform fees are applied **before** the creator-attributable pool. The manifest never includes protocol fee lines.

## V1 invariant (prospective rights)

Rights and economics version bumps on a `MediaAsset` restrict **future** usage/revenue only. Historical payouts are never clawed back. When referenced asset versions drift, posts move to `monetization_status = RESTRICTED` while `composition_status` may remain `VERIFIED`.

## Client sequence

### 1. Upload representation (off-chain)

- DripDrop: `POST /uploads/presign` → PUT bytes to R2.
- Compute `content_commitment` and `observed_fingerprint_commitment` (oracle-compatible hashing).

### 2. Request asset resolution (on-chain)

```text
media_asset::submit_media_resolution(
  content_commitment,
  observed_fingerprint_commitment,
  media_type,
  clock,
)
```

Emits `MediaResolutionRequestedEvent`. Oracle enqueues `resolve_media_asset`.

### 3. Oracle finalizes asset

Oracle calls `media_asset::finalize_media_asset` with:

- `claims[]` — verified authorship, rights-control, license-authority claims
- `usage_grants[]` — per-usage-class license terms (compensation, effective dates, revocability)
- `asset_kind` + optional `related_work_id` for musical composition ↔ sound recording links
- `beneficiary_splits[]` — economics resolution (beneficiary claims optional)

Returns canonical **`MediaAsset.id`** via `MediaAssetResolvedEvent`.

### 4. Create container (post or profile)

**Post:**

```text
post::create_post(..., media_asset_ids: vector<ID>, ...)
```

Emits `PostCreatedEvent` + `MediaAssetUsedEvent` per asset. Initial statuses: `composition_status = NONE`, `monetization_status = NONE`.

**Profile picture / cover:**

```text
profile::update_profile_picture_asset(profile, asset, display_url, clock, ctx)
profile::update_cover_photo_asset(profile, asset, display_url, clock, ctx)
```

Validates `UsageGrant` for `PROFILE_PICTURE` / `COVER_PHOTO`. Emits `MediaAssetUsedEvent` + `ProfileMediaAssetUpdatedEvent`.

### 5. Oracle composition analysis

gRPC sync enqueues `analyze_composition` when `PostCreatedEvent.media_asset_ids` is non-empty.

Oracle builds:

- `CompositionAnalysis` with `vector<AssetVersionCommitment>` (multi-asset)
- Optional `RevenueManifest` for derivative/mixed compositions
- `composition_status` + `monetization_status`

Submits:

```text
proof_of_creativity::analyze_post_composition(...)
proof_of_creativity::analyze_post_composition_sync_token_pool(...)  // when SPT pool linked
```

### 6. Revenue (layered)

On tips / SPT fees:

1. Extract protocol + platform fees from gross.
2. Apply `RevenueManifest` to **attributable pool only** when `monetization_status == ENABLED`.
3. Block manifest routing when stale (asset version mismatch → `monetization_status = RESTRICTED`).

## Python / oracle helpers

| Helper | Purpose |
|--------|---------|
| `build_submit_media_resolution_move_call` | Client upload step |
| `build_finalize_media_asset_move_call` | Oracle resolution |
| `build_composition_submission_from_assets` | Oracle composition + manifest |
| `build_analyze_post_composition_move_call` | Chain submission |

REST:

- `GET /oracle/assets/{id}` — resolved asset metadata
- `GET /oracle/assets/{id}/usages` — usage graph
- `GET /oracle/posts/{post_id}/manifest` — latest revenue manifest

## Deprecated (do not use for new work)

- Post-level `revenue_redirect_to` / `revenue_redirect_percentage`
- `analyze_and_update_post` / post-centric similarity badges as primary identity
- `enable_poc` on `PostCreatedEvent` (legacy decoder fallback only)
- Global `allow_monetization` on assets (replaced by per-`UsageGrant` compensation terms)

## Testing

```bash
pytest tests/integration/test_media_asset_e2e_flow.py tests/test_myso_submit_shape.py -q
```
