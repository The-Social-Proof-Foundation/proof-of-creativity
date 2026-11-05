"""Build Move transaction args for media_asset::finalize_media_asset."""

from __future__ import annotations

import os
import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Any

from nacl.exceptions import BadSignatureError
from nacl.signing import VerifyKey

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
LICENSE_ROYALTY_FREE = 3
COMPENSATION_NONE = 0
COMPENSATION_REVENUE_SHARE = 2
CLAIM_ASSERTED = 1
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
    grants = payload.get("usage_grants") or []
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


def default_claim(
    claim_type: int,
    claimant: str,
    rights_mask: int = 0,
    *,
    verification_status: int = CLAIM_ASSERTED,
    evidence_commitment: bytes | None = None,
) -> dict[str, Any]:
    return {
        "claim_type": claim_type,
        "claimant": claimant,
        "asset_id": None,
        "rights_mask": rights_mask,
        "scope": 0,
        "verification_status": verification_status,
        "evidence_commitment": evidence_commitment,
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
    # There is no safe universal license. Callers must supply evidence-backed grants.
    return []


def default_rights_payload(controller: str, *, now_ms: int = 0) -> dict[str, Any]:
    return {
        "claims": [],
        "usage_grants": [],
    }


def audio_rights_intent_hash(intent: dict[str, Any]) -> str:
    canonical = json.dumps(intent, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "0x" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _uleb128(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def verify_audio_rights_intent_signature(
    *, intent: dict[str, Any], wallet_signature: str, expected_owner: str,
) -> None:
    """Verify the Ed25519 MySo personal-message signature over the frozen intent hash."""
    try:
        serialized = base64.b64decode(wallet_signature, validate=True)
    except Exception as exc:
        raise ValueError("audio rights intent signature encoding is invalid") from exc
    if len(serialized) != 97 or serialized[0] != 0:
        raise ValueError("audio rights intent requires an Ed25519 user signature")
    signature = serialized[1:65]
    public_key = serialized[65:97]
    signer = "0x" + hashlib.blake2b(bytes([0]) + public_key, digest_size=32).hexdigest()
    if _normalize_object_id(signer) != _normalize_object_id(expected_owner):
        raise ValueError("audio rights intent signature does not match its owner")
    message = audio_rights_intent_hash(intent).encode("utf-8")
    bcs_personal_message = _uleb128(len(message)) + message
    digest = hashlib.blake2b(bytes([3, 0, 0]) + bcs_personal_message, digest_size=32).digest()
    try:
        VerifyKey(public_key).verify(digest, signature)
    except BadSignatureError as exc:
        raise ValueError("audio rights intent signature is invalid") from exc


def _assert_audio_provenance_policy(intent: dict[str, Any]) -> None:
    provenance = intent.get("provenance") or {}
    if int(provenance.get("policyVersion") or 0) != 1:
        raise ValueError("unsupported audio provenance policy")
    source = str(provenance.get("source") or "")
    basis = str(provenance.get("basis") or "")
    declaration = str(provenance.get("declaration") or "").strip()
    if source in ("user_upload", "post_original"):
        if basis != "original_rightsholder_attestation" or not declaration:
            raise ValueError("user audio requires a signed rightsholder attestation")
    elif source == "royalty_free_ingest":
        if basis != "licensed_catalog_evidence" or not declaration:
            raise ValueError("catalog audio requires licensed provenance evidence")
        if not provenance.get("evidenceUri") or not provenance.get("evidenceHash"):
            raise ValueError("catalog audio provenance evidence is incomplete")
    else:
        raise ValueError("unsupported audio provenance source")
    expires_raw = str(intent.get("expiresAt") or "").replace("Z", "+00:00")
    try:
        expires_at = datetime.fromisoformat(expires_raw)
    except ValueError as exc:
        raise ValueError("audio rights intent expiry is invalid") from exc
    if expires_at.tzinfo is None or expires_at <= datetime.now(timezone.utc):
        raise ValueError("audio rights intent is expired")


def audio_rights_from_intent(
    *,
    intent: dict[str, Any],
    wallet_signature: str,
    submitter: str,
    content_commitment_hex: str,
) -> dict[str, Any]:
    """Validate the frozen user intent and return the only grants allowed for v1 catalog audio."""
    owner = _normalize_object_id(str(intent.get("ownerWalletAddress") or ""))
    if owner != _normalize_object_id(submitter):
        raise ValueError("audio rights intent owner does not match chain submitter")
    expected_content = str(intent.get("contentCommitment") or "").lower().removeprefix("0x")
    if expected_content != content_commitment_hex.lower().removeprefix("0x"):
        raise ValueError("audio rights intent content commitment mismatch")
    if int(intent.get("version") or 0) != 1:
        raise ValueError("unsupported audio rights intent version")
    verify_audio_rights_intent_signature(
        intent=intent,
        wallet_signature=wallet_signature,
        expected_owner=owner,
    )
    _assert_audio_provenance_policy(intent)
    if str(intent.get("assetKind") or "") != "sound_recording":
        raise ValueError("standalone audio must resolve as sound_recording")

    grants: list[dict[str, Any]] = []
    for raw in intent.get("grants") or []:
        usage_class = int(raw.get("usageClass") or 0)
        if usage_class not in (USAGE_STREAM, USAGE_MUSIC_SOUNDTRACK):
            raise ValueError("audio catalog intent contains unsupported usage class")
        grant = {
            "usage_class": usage_class,
            "granted_rights": int(raw.get("grantedRights") or 0),
            "license_type": int(raw.get("licenseType") or LICENSE_ROYALTY_FREE),
            "compensation_type": int(raw.get("compensationType") or COMPENSATION_NONE),
            "compensation_bps": int(raw.get("compensationBps") or 0),
            "attribution_required": bool(raw.get("attributionRequired", True)),
            "derivatives_permitted": bool(raw.get("derivativesPermitted", False)),
            "commercial_use_permitted": bool(raw.get("commercialUsePermitted", False)),
            "effective_from": int(raw.get("effectiveFrom") or 0),
            "expires_at": raw.get("expiresAt"),
            "revocable": bool(raw.get("revocable", True)),
        }
        if grant["license_type"] != LICENSE_ROYALTY_FREE:
            raise ValueError("v1 audio catalog grants must be royalty-free")
        if grant["compensation_type"] != COMPENSATION_NONE or grant["compensation_bps"] != 0:
            raise ValueError("v1 audio catalog grants cannot configure compensation")
        if not grant_covers_usage(grant, usage_class):
            raise ValueError(f"audio grant does not cover usage class {usage_class}")
        grants.append(grant)
    if {int(grant["usage_class"]) for grant in grants} != {USAGE_STREAM, USAGE_MUSIC_SOUNDTRACK}:
        raise ValueError("audio catalog intent requires stream and soundtrack grants")

    evidence = hashlib.sha256(
        json.dumps(intent, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()
    controlled_rights = 0
    for grant in grants:
        controlled_rights |= int(grant["granted_rights"])
    claims = [
        default_claim(
            CLAIM_TYPE_AUTHORSHIP,
            owner,
            verification_status=CLAIM_ORACLE_VERIFIED,
            evidence_commitment=evidence,
        ),
        default_claim(
            CLAIM_TYPE_RIGHTS_CONTROL,
            owner,
            controlled_rights,
            verification_status=CLAIM_ORACLE_VERIFIED,
            evidence_commitment=evidence,
        ),
    ]
    return {"claims": claims, "usage_grants": grants, "evidence_commitment": evidence}


def verified_visual_registration_rights(
    *,
    submitter: str,
    content_commitment_hex: str,
    fingerprint_commitment_hex: str,
) -> dict[str, Any]:
    """Narrow legacy image/video registration rights backed by analyzed commitments."""
    evidence = hashlib.sha256(
        bytes.fromhex(content_commitment_hex.removeprefix("0x"))
        + bytes.fromhex(fingerprint_commitment_hex.removeprefix("0x"))
    ).digest()
    rights = required_rights_for_usage(USAGE_SOCIAL_POST)
    return {
        "claims": [
            default_claim(
                CLAIM_TYPE_AUTHORSHIP,
                submitter,
                verification_status=CLAIM_ORACLE_VERIFIED,
                evidence_commitment=evidence,
            ),
            default_claim(
                CLAIM_TYPE_RIGHTS_CONTROL,
                submitter,
                rights,
                verification_status=CLAIM_ORACLE_VERIFIED,
                evidence_commitment=evidence,
            ),
        ],
        "usage_grants": [
            {
                "usage_class": USAGE_SOCIAL_POST,
                "granted_rights": rights,
                "license_type": LICENSE_ROYALTY_FREE,
                "compensation_type": COMPENSATION_NONE,
                "compensation_bps": 0,
                "attribution_required": True,
                "derivatives_permitted": False,
                "commercial_use_permitted": False,
                "effective_from": 0,
                "expires_at": None,
                "revocable": True,
            }
        ],
    }


def default_beneficiary_splits(owner: str) -> list[dict[str, Any]]:
    return [{"beneficiary": owner, "share_bps": str(MANIFEST_BPS_TOTAL)}]


def _package_id() -> str:
    return os.getenv("MYSO_POC_PACKAGE_ID", "").strip()


def encode_beneficiary_split(beneficiary: str, share_bps: int) -> dict[str, Any]:
    """Field map for unsafe_moveCall JSON-to-BCS (no type wrapper)."""
    return {
        "beneficiary": _normalize_object_id(beneficiary),
        "share_bps": str(int(share_bps)),
    }


def encode_claim(claim: dict[str, Any]) -> dict[str, Any]:
    """Field map for unsafe_moveCall JSON-to-BCS (no type wrapper)."""
    evidence = claim.get("evidence_commitment")
    evidence_hex = bytes_to_move_hex(evidence) if isinstance(evidence, (bytes, bytearray)) else None
    asset_id = claim.get("asset_id")
    return {
        "claim_type": str(int(claim["claim_type"])),
        "claimant": _normalize_object_id(str(claim["claimant"])),
        "asset_id": _move_option_address(asset_id),
        "rights_mask": str(int(claim.get("rights_mask", 0))),
        "scope": str(int(claim.get("scope", 0))),
        "verification_status": str(int(claim.get("verification_status", CLAIM_ORACLE_VERIFIED))),
        "evidence_commitment": (
            {"Some": evidence_hex} if evidence_hex is not None else {"None": None}
        ),
    }


def encode_usage_grant(grant: dict[str, Any]) -> dict[str, Any]:
    """Field map for unsafe_moveCall JSON-to-BCS (no type wrapper)."""
    usage_class = int(grant["usage_class"])
    expires_at = grant.get("expires_at")
    return {
        "usage_class": str(usage_class),
        "granted_rights": str(int(grant.get("granted_rights", required_rights_for_usage(usage_class)))),
        "license_type": str(int(grant.get("license_type", LICENSE_NON_EXCLUSIVE))),
        "compensation_type": str(int(grant.get("compensation_type", COMPENSATION_NONE))),
        "compensation_bps": str(int(grant.get("compensation_bps", 0))),
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
    claims = resolution.claims
    usage_grants = resolution.usage_grants
    if not claims or not usage_grants:
        raise ValueError("finalize_media_asset requires evidence-backed claims and usage grants")

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


def build_update_media_asset_usage_grants_move_call(
    *,
    media_asset_id: str,
    usage_grants: list[dict[str, Any]],
    max_compensation_bps: int = 0,
    clock_id: str | None = None,
) -> dict[str, Any]:
    """Unsigned creator-wallet call for prospective usage-grant updates."""
    if not usage_grants:
        raise ValueError("usage_grants cannot be empty")
    return {
        "packageObjectId": _package_id(),
        "module": "media_asset",
        "function": "update_media_asset_usage_grants",
        "typeArguments": [],
        "arguments": [
            _normalize_object_id(media_asset_id),
            [encode_usage_grant(grant) for grant in usage_grants],
            str(int(max_compensation_bps)),
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
