"""Resolve a rights-dispute beneficiary at prepare/submit and hash that bundle.

The commitment is `sha3_256(bcs(RightsResolutionBundle))` for every kind, including
kind 0. Claimants are not rewritten after this. Implement recomputes the same hash.
"""

from __future__ import annotations

import re
from typing import Any

from app.chain.bcs_media_asset_claims import compute_rights_resolution_commitment
from app.chain.move_address import (
    IDENTITY_SOURCE_X,
    canonical_registry_username,
    identity_beneficiary_address,
    parse_identity_hash,
)
from app.services.myso_client import _normalize_object_id
from app.services.poc_chain_helpers import (
    fetch_profile_for_username,
    fetch_username_availability,
    fetch_username_beneficiary_by_identity,
    fetch_username_beneficiary_by_username,
    fetch_vault_beneficiary,
    get_network_profile,
)

TARGET_KIND_WALLET = 0
TARGET_KIND_VAULT = 1
TARGET_KIND_ONCHAIN_USERNAME = 2
TARGET_KIND_OFFCHAIN = 3

_USERNAME_RE = re.compile(r"^[a-z0-9_.]+$")


class RightsResolutionError(ValueError):
    """The dispute cannot be committed with this target."""


def sole_claimant(claims: list[dict[str, Any]]) -> str:
    if not claims:
        raise RightsResolutionError("rights dispute requires at least one claim")
    claimants = [_normalize_object_id(str(claim["claimant"])) for claim in claims]
    first = claimants[0]
    if any(item != first for item in claimants):
        raise RightsResolutionError(
            "rights dispute claimants must all equal one beneficiary address"
        )
    return first


def namespaced_offchain_username(handle: str) -> str:
    canonical = canonical_registry_username(handle)
    if not canonical or not _USERNAME_RE.fullmatch(canonical):
        raise RightsResolutionError(
            "off-chain handle must contain only a-z, 0-9, underscore, and dot"
        )
    name = "x_" + canonical
    if not 2 <= len(name) <= 50:
        raise RightsResolutionError("namespaced username must be 2-50 characters")
    return name


def _canonical_username(username: str) -> str:
    canonical = canonical_registry_username(username)
    if not canonical or not _USERNAME_RE.fullmatch(canonical):
        raise RightsResolutionError(
            "username must contain only a-z, 0-9, underscore, and dot"
        )
    return canonical


def _identity_hash_hex(identity_hash: str | bytes) -> str:
    return "0x" + parse_identity_hash(identity_hash).hex()


def _require_claims_match(claims: list[dict[str, Any]], beneficiary: str) -> None:
    claimant = sole_claimant(claims)
    if claimant != _normalize_object_id(beneficiary):
        raise RightsResolutionError(
            "claim claimants must equal the committed beneficiary address"
        )


def resolve_rights_target(
    network: str,
    claims: list[dict[str, Any]],
    target: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return the beneficiary and options that must be hashed and stored."""
    raw = target or {}
    kind = int(raw.get("kind") if raw.get("kind") is not None else raw.get("target_kind") or 0)
    chain = get_network_profile(network)

    if kind == TARGET_KIND_WALLET:
        beneficiary = sole_claimant(claims)
        return _bundle_target(
            target_kind=TARGET_KIND_WALLET,
            beneficiary_address=beneficiary,
            submit_function="submit_media_asset_rights_dispute_proposal",
        )

    if kind == TARGET_KIND_VAULT:
        vault_id = raw.get("vault_id")
        if not vault_id:
            raise RightsResolutionError("kind 1 requires vault_id")
        vault_id = _normalize_object_id(str(vault_id))
        beneficiary = fetch_vault_beneficiary(chain, vault_id)
        if not beneficiary:
            raise RightsResolutionError(f"PoCBeneficiaryVault {vault_id} was not found")
        beneficiary = _normalize_object_id(beneficiary)
        _require_claims_match(claims, beneficiary)
        return _bundle_target(
            target_kind=TARGET_KIND_VAULT,
            beneficiary_address=beneficiary,
            vault_id=vault_id,
            submit_function="submit_media_asset_rights_dispute_for_beneficiary",
        )

    if kind == TARGET_KIND_ONCHAIN_USERNAME:
        username_raw = raw.get("username")
        if not username_raw:
            raise RightsResolutionError("kind 2 requires username")
        username = _canonical_username(str(username_raw))
        existing = fetch_username_beneficiary_by_username(chain, username)
        if existing and existing.get("beneficiaryAddress"):
            beneficiary = _normalize_object_id(str(existing["beneficiaryAddress"]))
            committed_username = str(existing.get("username") or username)
            _require_claims_match(claims, beneficiary)
            return _bundle_target(
                target_kind=TARGET_KIND_ONCHAIN_USERNAME,
                beneficiary_address=beneficiary,
                username=committed_username,
                submit_function="submit_media_asset_rights_dispute_for_onchain_username",
            )
        profile = fetch_profile_for_username(chain, username)
        if not profile:
            raise RightsResolutionError(
                "kind 2 requires an on-chain username beneficiary or a registered profile"
            )
        beneficiary = _normalize_object_id(profile["owner"])
        _require_claims_match(claims, beneficiary)
        return _bundle_target(
            target_kind=TARGET_KIND_ONCHAIN_USERNAME,
            beneficiary_address=beneficiary,
            username=username,
            submit_function="submit_media_asset_rights_dispute_for_profile_username",
        )

    if kind == TARGET_KIND_OFFCHAIN:
        handle = raw.get("handle") or raw.get("username")
        identity_hash = raw.get("identity_hash")
        if not handle or not identity_hash:
            raise RightsResolutionError("kind 3 requires handle and identity_hash")
        source = raw.get("identity_source")
        source = IDENTITY_SOURCE_X if source is None else int(source)
        hash_hex = _identity_hash_hex(identity_hash)
        existing = fetch_username_beneficiary_by_identity(
            chain, hash_hex, identity_source=source
        )
        if existing and existing.get("beneficiaryAddress") and existing.get("username"):
            beneficiary = _normalize_object_id(str(existing["beneficiaryAddress"]))
            _require_claims_match(claims, beneficiary)
            return _bundle_target(
                target_kind=TARGET_KIND_OFFCHAIN,
                beneficiary_address=beneficiary,
                username=str(existing["username"]),
                identity_source=source,
                identity_hash=hash_hex,
                submit_function="submit_media_asset_rights_dispute_for_offchain_identity",
            )
        username = namespaced_offchain_username(str(handle))
        if not fetch_username_availability(chain, username):
            raise RightsResolutionError(f"namespaced username {username} is reserved or taken")
        beneficiary = _normalize_object_id(
            identity_beneficiary_address(hash_hex, identity_source=source)
        )
        _require_claims_match(claims, beneficiary)
        return _bundle_target(
            target_kind=TARGET_KIND_OFFCHAIN,
            beneficiary_address=beneficiary,
            username=username,
            identity_source=source,
            identity_hash=hash_hex,
            submit_function="submit_media_asset_rights_dispute_for_new_offchain_identity",
        )

    raise RightsResolutionError(f"unknown rights target kind {kind}")


def commitment_for_resolution(
    claims: list[dict[str, Any]],
    usage_grants: list[dict[str, Any]],
    resolved: dict[str, Any],
) -> bytes:
    return compute_rights_resolution_commitment(claims, usage_grants, resolved)


def target_from_bundle_row(row: dict[str, Any]) -> dict[str, Any]:
    source = row.get("identity_source")
    return {
        "target_kind": int(row.get("target_kind") or 0),
        "beneficiary_address": str(row.get("beneficiary_address") or "0x0"),
        "vault_id": row.get("target_vault_id"),
        "username": row.get("target_username"),
        "identity_source": int(source) if source is not None else None,
        "identity_hash": row.get("identity_hash"),
    }


def _bundle_target(
    *,
    target_kind: int,
    beneficiary_address: str,
    submit_function: str,
    vault_id: str | None = None,
    username: str | None = None,
    identity_source: int | None = None,
    identity_hash: str | None = None,
) -> dict[str, Any]:
    return {
        "target_kind": target_kind,
        "beneficiary_address": beneficiary_address,
        "vault_id": vault_id,
        "username": username,
        "identity_source": identity_source,
        "identity_hash": identity_hash,
        "submit_function": submit_function,
    }
