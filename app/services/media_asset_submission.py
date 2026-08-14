"""Build Move transaction args for media_asset::finalize_media_asset."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from app.chain.move_address import bytes_to_move_hex
from app.services.myso_client import _clock_object_id, _move_option_address, _normalize_object_id

# Mirrors social_contracts::media_asset
ORIGINALITY_ORIGINAL = 1
ORIGINALITY_DERIVATIVE = 2
MANIFEST_BPS_TOTAL = 10_000

RIGHT_REPRODUCTION = 1 << 0
RIGHT_DERIVATIVE_WORK = 1 << 1
RIGHT_DISTRIBUTION = 1 << 2
RIGHT_PUBLIC_PERFORMANCE = 1 << 3
RIGHT_PUBLIC_DISPLAY = 1 << 4
RIGHT_DIGITAL_AUDIO_TRANSMISSION = 1 << 5
ALL_STATUTORY_RIGHTS = (
    RIGHT_REPRODUCTION
    | RIGHT_DERIVATIVE_WORK
    | RIGHT_DISTRIBUTION
    | RIGHT_PUBLIC_PERFORMANCE
    | RIGHT_PUBLIC_DISPLAY
    | RIGHT_DIGITAL_AUDIO_TRANSMISSION
)

USAGE_SOCIAL_POST = 1
USAGE_PROFILE_PICTURE = 2
USAGE_COVER_PHOTO = 3
USAGE_ARTICLE_EMBED = 4
USAGE_MUSIC_SOUNDTRACK = 5
USAGE_ADVERTISEMENT = 6
USAGE_REMIX = 7
USAGE_DOWNLOAD = 8
USAGE_STREAM = 9

LICENSE_NON_EXCLUSIVE = 1
COMPENSATION_REVENUE_SHARE = 2
CLAIM_ORACLE_VERIFIED = 2

CLAIM_TYPE_AUTHORSHIP = 1
CLAIM_TYPE_COPYRIGHT_OWNERSHIP = 2
CLAIM_TYPE_RIGHTS_CONTROL = 3
CLAIM_TYPE_LICENSE_AUTHORITY = 4
CLAIM_TYPE_BENEFICIARY = 5

ASSET_KIND_UNSPECIFIED = 0
ASSET_KIND_VISUAL_WORK = 1
ASSET_KIND_MUSICAL_COMPOSITION = 2
ASSET_KIND_SOUND_RECORDING = 3


def required_rights_for_usage(usage_class: int) -> int:
    if usage_class == USAGE_SOCIAL_POST:
        return RIGHT_REPRODUCTION | RIGHT_PUBLIC_DISPLAY
    if usage_class in (USAGE_PROFILE_PICTURE, USAGE_COVER_PHOTO):
        return RIGHT_REPRODUCTION | RIGHT_PUBLIC_DISPLAY
    if usage_class == USAGE_ARTICLE_EMBED:
        return RIGHT_REPRODUCTION | RIGHT_PUBLIC_DISPLAY | RIGHT_DISTRIBUTION
    if usage_class == USAGE_MUSIC_SOUNDTRACK:
        return (
            RIGHT_REPRODUCTION
            | RIGHT_DERIVATIVE_WORK
            | RIGHT_PUBLIC_PERFORMANCE
            | RIGHT_DIGITAL_AUDIO_TRANSMISSION
        )
    if usage_class == USAGE_ADVERTISEMENT:
        return (
            RIGHT_REPRODUCTION
            | RIGHT_PUBLIC_DISPLAY
            | RIGHT_PUBLIC_PERFORMANCE
            | RIGHT_DISTRIBUTION
        )
    if usage_class == USAGE_REMIX:
        return RIGHT_DERIVATIVE_WORK | RIGHT_REPRODUCTION
    if usage_class == USAGE_DOWNLOAD:
        return RIGHT_REPRODUCTION | RIGHT_DISTRIBUTION
    if usage_class == USAGE_STREAM:
        return (
            RIGHT_REPRODUCTION
            | RIGHT_PUBLIC_PERFORMANCE
            | RIGHT_DIGITAL_AUDIO_TRANSMISSION
        )
    return 0


def _usage_requires_derivatives(usage_class: int) -> bool:
    return usage_class in (USAGE_REMIX, USAGE_MUSIC_SOUNDTRACK)


def _usage_requires_commercial(usage_class: int) -> bool:
    return usage_class == USAGE_ADVERTISEMENT


def grant_is_effective(grant: dict[str, Any], now_ms: int) -> bool:
    effective_from = int(grant.get("effective_from", 0))
    if now_ms < effective_from:
        return False
    expires_at = grant.get("expires_at")
    if expires_at is not None:
        if now_ms >= int(expires_at):
            return False
    return True


def grant_covers_usage(grant: dict[str, Any], usage_class: int, *, now_ms: int = 0) -> bool:
    if int(grant.get("usage_class", 0)) != usage_class:
        return False
    if not grant_is_effective(grant, now_ms):
        return False
    granted = int(grant.get("granted_rights", 0))
    required = required_rights_for_usage(usage_class)
    if (granted & required) != required:
        return False
    if _usage_requires_derivatives(usage_class) and not bool(grant.get("derivatives_permitted")):
        return False
    if _usage_requires_commercial(usage_class) and not bool(grant.get("commercial_use_permitted")):
        return False
    return True


def usage_permitted(rights_payload: dict[str, Any] | None, usage_class: int, *, now_ms: int = 0) -> bool:
    """True when rights_json contains an effective UsageGrant covering the usage class."""
    payload = rights_payload or {}
    grants = payload.get("usage_grants") or default_usage_grants(now_ms)
    for grant in grants:
        if grant_covers_usage(grant, usage_class, now_ms=now_ms):
            return True
    return False


@dataclass
class MediaResolutionResult:
    """Oracle resolution outcome for a MediaResolutionRequest."""

    request_id: str
    content_commitment: bytes
    observed_fingerprint_commitment: bytes
    media_type: int
    submitter: str
    link_to_existing_id: str | None = None
    originality_status: int = ORIGINALITY_ORIGINAL
    lineage_parent_id: str | None = None
    asset_kind: int = ASSET_KIND_UNSPECIFIED
    related_work_id: str | None = None
    claims: list[dict[str, Any]] = field(default_factory=list)
    usage_grants: list[dict[str, Any]] = field(default_factory=list)
    beneficiary_splits: list[dict[str, Any]] = field(default_factory=list)
    dedup_matched: bool = False


def default_claim(claim_type: int, claimant: str, rights_mask: int = 0) -> dict[str, Any]:
    return {
        "claim_type": claim_type,
        "claimant": claimant,
        "asset_id": None,
        "rights_mask": rights_mask,
        "scope": 0,
        "verification_status": CLAIM_ORACLE_VERIFIED,
        "evidence_commitment": None,
    }


def default_claims(owner: str) -> list[dict[str, Any]]:
    return [
        default_claim(CLAIM_TYPE_AUTHORSHIP, owner),
        default_claim(CLAIM_TYPE_RIGHTS_CONTROL, owner, ALL_STATUTORY_RIGHTS),
        default_claim(CLAIM_TYPE_LICENSE_AUTHORITY, owner),
    ]


def default_usage_grant(
    usage_class: int,
    effective_from: int = 0,
    *,
    max_compensation_bps: int | None = None,
) -> dict[str, Any]:
    cap = MANIFEST_BPS_TOTAL
    if max_compensation_bps is not None:
        cap = min(int(max_compensation_bps), MANIFEST_BPS_TOTAL)
    return {
        "usage_class": usage_class,
        "granted_rights": required_rights_for_usage(usage_class),
        "license_type": LICENSE_NON_EXCLUSIVE,
        "compensation_type": COMPENSATION_REVENUE_SHARE,
        "compensation_bps": cap,
        "attribution_required": False,
        "derivatives_permitted": True,
        "commercial_use_permitted": True,
        "effective_from": effective_from,
        "expires_at": None,
        "revocable": True,
    }


def default_usage_grants(effective_from: int = 0) -> list[dict[str, Any]]:
    return [default_usage_grant(u, effective_from) for u in range(1, 10)]


def default_rights_payload(controller: str, *, now_ms: int = 0) -> dict[str, Any]:
    return {
        "claims": default_claims(controller),
        "usage_grants": default_usage_grants(now_ms),
    }


def default_beneficiary_splits(owner: str) -> list[dict[str, Any]]:
    return [{"beneficiary": owner, "share_bps": str(MANIFEST_BPS_TOTAL)}]


def _package_id() -> str:
    return os.getenv("MYSO_POC_PACKAGE_ID", "").strip()


def _struct_type(name: str) -> str:
    pkg = _package_id()
    return f"{pkg}::media_asset::{name}"


def encode_beneficiary_split(beneficiary: str, share_bps: int) -> dict[str, Any]:
    return {
        "type": _struct_type("BeneficiarySplit"),
        "fields": {
            "beneficiary": _normalize_object_id(beneficiary),
            "share_bps": str(int(share_bps)),
        },
    }


def encode_claim(claim: dict[str, Any]) -> dict[str, Any]:
    evidence = claim.get("evidence_commitment")
    evidence_hex = bytes_to_move_hex(evidence) if isinstance(evidence, (bytes, bytearray)) else None
    asset_id = claim.get("asset_id")
    return {
        "type": _struct_type("Claim"),
        "fields": {
            "claim_type": str(int(claim["claim_type"])),
            "claimant": _normalize_object_id(str(claim["claimant"])),
            "asset_id": _move_option_address(asset_id),
            "rights_mask": str(int(claim.get("rights_mask", 0))),
            "scope": str(int(claim.get("scope", 0))),
            "verification_status": str(int(claim.get("verification_status", CLAIM_ORACLE_VERIFIED))),
            "evidence_commitment": (
                {"Some": evidence_hex} if evidence_hex is not None else {"None": None}
            ),
        },
    }


def encode_usage_grant(grant: dict[str, Any]) -> dict[str, Any]:
    usage_class = int(grant["usage_class"])
    expires_at = grant.get("expires_at")
    return {
        "type": _struct_type("UsageGrant"),
        "fields": {
            "usage_class": str(usage_class),
            "granted_rights": str(int(grant.get("granted_rights", required_rights_for_usage(usage_class)))),
            "license_type": str(int(grant.get("license_type", LICENSE_NON_EXCLUSIVE))),
            "compensation_type": str(int(grant.get("compensation_type", COMPENSATION_REVENUE_SHARE))),
            "compensation_bps": str(int(grant.get("compensation_bps", MANIFEST_BPS_TOTAL))),
            "attribution_required": bool(grant.get("attribution_required", False)),
            "derivatives_permitted": bool(
                grant.get("derivatives_permitted", _usage_requires_derivatives(usage_class) or True)
            ),
            "commercial_use_permitted": bool(grant.get("commercial_use_permitted", True)),
            "effective_from": str(int(grant.get("effective_from", 0))),
            "expires_at": (
                {"Some": str(int(expires_at))} if expires_at is not None else {"None": None}
            ),
            "revocable": bool(grant.get("revocable", True)),
        },
    }


def build_finalize_media_asset_move_call(
    resolution: MediaResolutionResult,
    *,
    config_id: str | None = None,
    clock_id: str | None = None,
    now_ms: int = 0,
) -> dict[str, Any]:
    """Return unsafe_moveCall payload for media_asset::finalize_media_asset."""
    cfg = config_id or os.getenv("MYSO_POC_CONFIG_ID", "")
    if not cfg:
        raise RuntimeError("MYSO_POC_CONFIG_ID not configured")

    splits = resolution.beneficiary_splits
    if not splits:
        splits = default_beneficiary_splits(resolution.submitter)
    encoded_splits = [
        encode_beneficiary_split(str(s["beneficiary"]), int(s.get("share_bps", MANIFEST_BPS_TOTAL)))
        for s in splits
    ]

    primary = _normalize_object_id(resolution.submitter)
    claims = resolution.claims or default_claims(primary)
    usage_grants = resolution.usage_grants or default_usage_grants(now_ms)

    return {
        "packageObjectId": _package_id(),
        "module": "proof_of_creativity",
        "function": "finalize_media_asset",
        "typeArguments": [],
        "arguments": [
            cfg,
            _normalize_object_id(resolution.request_id),
            _move_option_address(resolution.link_to_existing_id),
            int(resolution.originality_status),
            _move_option_address(resolution.lineage_parent_id),
            int(resolution.asset_kind),
            _move_option_address(resolution.related_work_id),
            [encode_claim(c) for c in claims],
            [encode_usage_grant(g) for g in usage_grants],
            encoded_splits,
            clock_id or _clock_object_id(),
        ],
    }


def build_submit_media_resolution_move_call(
    *,
    content_commitment: bytes,
    observed_fingerprint_commitment: bytes,
    media_type: int,
    clock_id: str | None = None,
) -> dict[str, Any]:
    """Client/oracle: request PoC resolution for an uploaded representation."""
    return {
        "packageObjectId": _package_id(),
        "module": "media_asset",
        "function": "submit_media_resolution",
        "typeArguments": [],
        "arguments": [
            bytes_to_move_hex(content_commitment),
            bytes_to_move_hex(observed_fingerprint_commitment),
            str(int(media_type)),
            clock_id or _clock_object_id(),
        ],
    }


def build_finalize_media_asset_from_dedup(
    *,
    request_id: str,
    content_commitment: bytes,
    observed_fingerprint_commitment: bytes,
    media_type: int,
    submitter: str,
    existing_asset_id: str,
    originality_status: int = ORIGINALITY_DERIVATIVE,
    lineage_parent_id: str | None = None,
) -> dict[str, Any]:
    """Link fingerprint to an existing canonical asset (dedup hit)."""
    resolution = MediaResolutionResult(
        request_id=request_id,
        content_commitment=content_commitment,
        observed_fingerprint_commitment=observed_fingerprint_commitment,
        media_type=media_type,
        submitter=submitter,
        link_to_existing_id=existing_asset_id,
        originality_status=originality_status,
        lineage_parent_id=lineage_parent_id or existing_asset_id,
        dedup_matched=True,
    )
    return build_finalize_media_asset_move_call(resolution)


def commitment_hex(data: bytes) -> str:
    return bytes_to_move_hex(data)
