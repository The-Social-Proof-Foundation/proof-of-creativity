"""Build MoveCall payloads for proof_of_creativity username beneficiary entries."""

from __future__ import annotations

from typing import Any

from app.chain.move_address import (
    IDENTITY_SOURCE_X,
    bytes_to_move_hex,
    canonical_registry_username,
    text_to_move_hex,
)


def _require(obj_id: str | None, name: str) -> str:
    if not obj_id or not str(obj_id).strip():
        raise RuntimeError(f"Missing on-chain object id for {name}")
    return str(obj_id).strip()


def build_create_username_beneficiary_call(
    *,
    package_id: str,
    admin_cap_id: str,
    directory_id: str,
    shard_id: str,
    vault_directory_id: str,
    username_registry_id: str,
    username: str,
    identity_hash: str | bytes,
    required_x_handle: str | None = None,
    clock_id: str,
    identity_source: int = IDENTITY_SOURCE_X,
) -> dict[str, Any]:
    handle = required_x_handle or username
    canonical_user = canonical_registry_username(username)
    canonical_handle = canonical_registry_username(handle)
    ih = identity_hash if isinstance(identity_hash, bytes) else identity_hash
    if isinstance(ih, str):
        from app.chain.move_address import parse_identity_hash

        ih_bytes = parse_identity_hash(ih)
    else:
        ih_bytes = ih

    return {
        "packageObjectId": package_id,
        "module": "proof_of_creativity",
        "function": "create_username_beneficiary",
        "typeArguments": [],
        "arguments": [
            _require(admin_cap_id, "PoCBeneficiaryAdminCap"),
            _require(directory_id, "PoCUsernameBeneficiaryDirectory"),
            _require(shard_id, "PoCUsernameBeneficiaryShard"),
            _require(vault_directory_id, "PoCVaultDirectory"),
            _require(username_registry_id, "UsernameRegistry"),
            text_to_move_hex(canonical_user),
            identity_source,
            bytes_to_move_hex(ih_bytes),
            text_to_move_hex(canonical_handle),
            _require(clock_id, "Clock"),
        ],
    }


def build_claim_username_beneficiary_call(
    *,
    package_id: str,
    poc_config_id: str,
    profile_config_id: str,
    directory_id: str,
    shard_id: str,
    username_registry_id: str,
    memory_registry_id: str,
    ai_credit_config_id: str,
    beneficiary_id: str,
    evidence_hash: bytes,
    attested_x_handle: str,
    display_name: str,
    bio: str,
    profile_picture_url: str,
    cover_photo_url: str,
    wallet: str,
    clock_id: str,
) -> dict[str, Any]:
    handle = canonical_registry_username(attested_x_handle)
    return {
        "packageObjectId": package_id,
        "module": "proof_of_creativity",
        "function": "claim_username_beneficiary",
        "typeArguments": [],
        "arguments": [
            _require(poc_config_id, "PoCConfig"),
            _require(profile_config_id, "ProfileConfig"),
            _require(directory_id, "PoCUsernameBeneficiaryDirectory"),
            _require(shard_id, "PoCUsernameBeneficiaryShard"),
            _require(username_registry_id, "UsernameRegistry"),
            _require(memory_registry_id, "MemoryRegistry"),
            _require(ai_credit_config_id, "AiCreditConfig"),
            _require(beneficiary_id, "PoCUsernameBeneficiary"),
            bytes_to_move_hex(evidence_hash),
            text_to_move_hex(handle),
            text_to_move_hex(display_name),
            text_to_move_hex(bio),
            text_to_move_hex(profile_picture_url),
            text_to_move_hex(cover_photo_url),
            _require(wallet, "wallet"),
            _require(clock_id, "Clock"),
        ],
    }
