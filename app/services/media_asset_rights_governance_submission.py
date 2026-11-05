"""PTB / MoveCall builders for media asset rights governance disputes."""

from __future__ import annotations

import os
from typing import Any

from app.chain.move_address import bytes_to_move_hex
from app.services.media_asset_submission import encode_claim, encode_usage_grant
from app.services.myso_client import _clock_object_id, _normalize_object_id


def _package_id() -> str:
    return os.getenv("MYSO_POC_PACKAGE_ID", "").strip()


def _poc_config_id() -> str:
    return os.getenv("MYSO_POC_CONFIG_ID", "").strip()


def _governance_registry_id() -> str:
    return os.getenv("POC_GOVERNANCE_REGISTRY_ID", "").strip()


def _ecosystem_treasury_id() -> str:
    return os.getenv("MYSO_ECOSYSTEM_TREASURY_ID", "").strip()


def build_finalize_media_asset_rights_governance_move_call(
    *,
    proposal_id: str,
    media_asset_id: str,
    config_id: str | None = None,
    governance_registry_id: str | None = None,
    treasury_id: str | None = None,
    clock_id: str | None = None,
) -> dict[str, Any]:
    return {
        "packageObjectId": _package_id(),
        "module": "proof_of_creativity",
        "function": "finalize_media_asset_rights_governance_proposal",
        "typeArguments": [],
        "arguments": [
            _normalize_object_id(config_id or _poc_config_id()),
            _normalize_object_id(governance_registry_id or _governance_registry_id()),
            _normalize_object_id(proposal_id),
            _normalize_object_id(media_asset_id),
            _normalize_object_id(treasury_id or _ecosystem_treasury_id()),
            _normalize_object_id(clock_id or _clock_object_id()),
        ],
    }


def _object_id(profile: Any, env_key: str, profile_key: str, label: str) -> str:
    val = os.getenv(env_key, "").strip()
    if not val and profile is not None:
        val = str((getattr(profile, "objects", None) or {}).get(profile_key) or "").strip()
    if not val:
        raise RuntimeError(f"Missing on-chain object id for {label} ({env_key})")
    return _normalize_object_id(val)


def _rights_call(
    profile: Any,
    function: str,
    arguments: list[Any],
) -> dict[str, Any]:
    package = _package_id()
    if not package and profile is not None:
        package = str((getattr(profile, "objects", None) or {}).get("poc_package") or "").strip()
    return {
        "packageObjectId": package,
        "module": "proof_of_creativity",
        "function": function,
        "typeArguments": [],
        "arguments": arguments,
    }


def _shared_rights_prefix(
    profile: Any,
    *,
    proposal_id: str,
    media_asset_id: str,
    config_id: str | None,
    governance_registry_id: str | None,
) -> list[Any]:
    _ = profile
    config = config_id or _poc_config_id()
    registry = governance_registry_id or _governance_registry_id()
    if not config or not registry:
        raise RuntimeError("PoC config and governance registry ids are required")
    return [
        _normalize_object_id(config),
        _normalize_object_id(registry),
        _normalize_object_id(proposal_id),
        _normalize_object_id(media_asset_id),
    ]


def build_implement_media_asset_rights_to_beneficiary_move_call(
    *,
    profile: Any,
    proposal_id: str,
    media_asset_id: str,
    vault_id: str,
    claims: list[dict[str, Any]],
    usage_grants: list[dict[str, Any]],
    reasoning: str,
    config_id: str | None = None,
    governance_registry_id: str | None = None,
    clock_id: str | None = None,
) -> dict[str, Any]:
    encoded_claims = [encode_claim(c) for c in claims]
    encoded_grants = [encode_usage_grant(g) for g in usage_grants]
    return _rights_call(
        profile,
        "implement_media_asset_rights_to_beneficiary",
        [
            *_shared_rights_prefix(
                profile,
                proposal_id=proposal_id,
                media_asset_id=media_asset_id,
                config_id=config_id,
                governance_registry_id=governance_registry_id,
            ),
            _normalize_object_id(vault_id),
            encoded_claims,
            encoded_grants,
            reasoning,
            _normalize_object_id(clock_id or _clock_object_id()),
        ],
    )


def build_implement_media_asset_rights_to_onchain_username_move_call(
    *,
    profile: Any,
    proposal_id: str,
    media_asset_id: str,
    beneficiary_id: str,
    username: str,
    claims: list[dict[str, Any]],
    usage_grants: list[dict[str, Any]],
    reasoning: str,
    config_id: str | None = None,
    governance_registry_id: str | None = None,
    clock_id: str | None = None,
) -> dict[str, Any]:
    from app.services.poc_chain_helpers import resolve_username_beneficiary_shard_id

    directory_id = _object_id(
        profile,
        "MYSO_POC_USERNAME_BENEFICIARY_DIRECTORY_ID",
        "username_beneficiary_directory",
        "PoCUsernameBeneficiaryDirectory",
    )
    shard_id = resolve_username_beneficiary_shard_id(
        profile, username, directory_id=directory_id
    )
    encoded_claims = [encode_claim(c) for c in claims]
    encoded_grants = [encode_usage_grant(g) for g in usage_grants]
    return _rights_call(
        profile,
        "implement_media_asset_rights_to_onchain_username",
        [
            *_shared_rights_prefix(
                profile,
                proposal_id=proposal_id,
                media_asset_id=media_asset_id,
                config_id=config_id,
                governance_registry_id=governance_registry_id,
            ),
            directory_id,
            _normalize_object_id(shard_id),
            _normalize_object_id(beneficiary_id),
            encoded_claims,
            encoded_grants,
            reasoning,
            _normalize_object_id(clock_id or _clock_object_id()),
        ],
    )


def build_implement_media_asset_rights_to_profile_username_move_call(
    *,
    profile: Any,
    proposal_id: str,
    media_asset_id: str,
    profile_id: str,
    username: str,
    claims: list[dict[str, Any]],
    usage_grants: list[dict[str, Any]],
    reasoning: str,
    config_id: str | None = None,
    governance_registry_id: str | None = None,
    clock_id: str | None = None,
) -> dict[str, Any]:
    from app.services.poc_chain_helpers import resolve_username_beneficiary_shard_id

    directory_id = _object_id(
        profile,
        "MYSO_POC_USERNAME_BENEFICIARY_DIRECTORY_ID",
        "username_beneficiary_directory",
        "PoCUsernameBeneficiaryDirectory",
    )
    shard_id = resolve_username_beneficiary_shard_id(
        profile, username, directory_id=directory_id
    )
    encoded_claims = [encode_claim(c) for c in claims]
    encoded_grants = [encode_usage_grant(g) for g in usage_grants]
    return _rights_call(
        profile,
        "implement_media_asset_rights_to_profile_username",
        [
            *_shared_rights_prefix(
                profile,
                proposal_id=proposal_id,
                media_asset_id=media_asset_id,
                config_id=config_id,
                governance_registry_id=governance_registry_id,
            ),
            _normalize_object_id(shard_id),
            _object_id(
                profile, "MYSO_USERNAME_REGISTRY_ID", "username_registry", "UsernameRegistry"
            ),
            _normalize_object_id(profile_id),
            encoded_claims,
            encoded_grants,
            reasoning,
            _normalize_object_id(clock_id or _clock_object_id()),
        ],
    )


def build_implement_media_asset_rights_to_offchain_identity_move_call(
    *,
    profile: Any,
    proposal_id: str,
    media_asset_id: str,
    beneficiary_id: str,
    username: str,
    claims: list[dict[str, Any]],
    usage_grants: list[dict[str, Any]],
    reasoning: str,
    config_id: str | None = None,
    governance_registry_id: str | None = None,
    clock_id: str | None = None,
) -> dict[str, Any]:
    from app.services.poc_chain_helpers import resolve_username_beneficiary_shard_id

    directory_id = _object_id(
        profile,
        "MYSO_POC_USERNAME_BENEFICIARY_DIRECTORY_ID",
        "username_beneficiary_directory",
        "PoCUsernameBeneficiaryDirectory",
    )
    shard_id = resolve_username_beneficiary_shard_id(
        profile, username, directory_id=directory_id
    )
    encoded_claims = [encode_claim(c) for c in claims]
    encoded_grants = [encode_usage_grant(g) for g in usage_grants]
    return _rights_call(
        profile,
        "implement_media_asset_rights_to_offchain_identity",
        [
            *_shared_rights_prefix(
                profile,
                proposal_id=proposal_id,
                media_asset_id=media_asset_id,
                config_id=config_id,
                governance_registry_id=governance_registry_id,
            ),
            directory_id,
            _normalize_object_id(shard_id),
            _normalize_object_id(beneficiary_id),
            encoded_claims,
            encoded_grants,
            reasoning,
            _normalize_object_id(clock_id or _clock_object_id()),
        ],
    )


def build_implement_media_asset_rights_move_call(
    *,
    proposal_id: str,
    media_asset_id: str,
    claims: list[dict[str, Any]],
    usage_grants: list[dict[str, Any]],
    reasoning: str,
    evidence_urls: list[str] | None = None,
    config_id: str | None = None,
    governance_registry_id: str | None = None,
    treasury_id: str | None = None,
    clock_id: str | None = None,
) -> dict[str, Any]:
    encoded_claims = [encode_claim(c) for c in claims]
    encoded_grants = [encode_usage_grant(g) for g in usage_grants]
    evidence = {"None": None}
    if evidence_urls:
        evidence = {"Some": evidence_urls}
    return {
        "packageObjectId": _package_id(),
        "module": "proof_of_creativity",
        "function": "finalize_media_asset_rights_via_dao",
        "typeArguments": [],
        "arguments": [
            _normalize_object_id(config_id or _poc_config_id()),
            _normalize_object_id(governance_registry_id or _governance_registry_id()),
            _normalize_object_id(proposal_id),
            _normalize_object_id(media_asset_id),
            _normalize_object_id(treasury_id or _ecosystem_treasury_id()),
            encoded_claims,
            encoded_grants,
            reasoning,
            evidence,
            _normalize_object_id(clock_id or _clock_object_id()),
        ],
    }


def build_submit_media_asset_rights_dispute_move_call(
    *,
    media_asset_id: str,
    title: str,
    description: str,
    claims_commitment: bytes,
    payment_coin: str,
    config_id: str | None = None,
    governance_registry_id: str | None = None,
    treasury_id: str | None = None,
    clock_id: str | None = None,
    metadata_json: str | None = None,
) -> dict[str, Any]:
    return {
        "packageObjectId": _package_id(),
        "module": "proof_of_creativity",
        "function": "submit_media_asset_rights_dispute_proposal",
        "typeArguments": [],
        "arguments": [
            _normalize_object_id(config_id or _poc_config_id()),
            _normalize_object_id(governance_registry_id or _governance_registry_id()),
            _normalize_object_id(treasury_id or _ecosystem_treasury_id()),
            _normalize_object_id(media_asset_id),
            title,
            description,
            bytes_to_move_hex(claims_commitment),
            {"None": None},
            {"None": None},
            {"Some": metadata_json} if metadata_json else {"None": None},
            payment_coin,
            _normalize_object_id(clock_id or _clock_object_id()),
        ],
    }
