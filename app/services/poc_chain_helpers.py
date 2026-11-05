"""GraphQL/RPC helpers for PoC vault, shards, and username beneficiaries."""

from __future__ import annotations

import json
from typing import Any

import requests
import structlog

from app.chain.graphql_session import resolve_graphql_url
from app.chain.move_address import shard_index_for_username
from app.network_config import NetworkProfile, load_network_profile

logger = structlog.get_logger()

NUM_SHARDS = 256

_BENEFICIARY_STATUS_ACTIVE = frozenset({None, "ACTIVE", 0, "0", 1, "1"})
_BENEFICIARY_STATUS_CLAIMED = frozenset({2, "2", "CLAIMED"})
_BENEFICIARY_STATUS_ENDED = frozenset({3, "3", "ENDED"})


def normalize_beneficiary_status(raw: Any) -> str | None:
    """Map Move UsernameBeneficiary.status variants to ACTIVE | CLAIMED | ENDED."""
    if raw in _BENEFICIARY_STATUS_ACTIVE:
        return "ACTIVE"
    if raw in _BENEFICIARY_STATUS_CLAIMED:
        return "CLAIMED"
    if raw in _BENEFICIARY_STATUS_ENDED:
        return "ENDED"
    return str(raw)


def _graphql_post(profile: NetworkProfile, query: str, variables: dict | None = None) -> dict[str, Any]:
    url = resolve_graphql_url(profile)
    payload: dict[str, Any] = {"query": query}
    if variables:
        payload["variables"] = variables
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=15.0,
    )
    response.raise_for_status()
    body = response.json()
    if body.get("errors"):
        raise RuntimeError(f"GraphQL errors: {body['errors']}")
    return body.get("data") or {}


def _rpc_get_object_fields(rpc_url: str, object_id: str) -> dict[str, Any]:
    response = requests.post(
        rpc_url,
        headers={"Content-Type": "application/json"},
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "myso_getObject",
            "params": [object_id, {"showContent": True}],
        },
        timeout=10.0,
    )
    response.raise_for_status()
    result = response.json()
    if result.get("error"):
        raise RuntimeError(str(result["error"]))
    return (
        result.get("result", {})
        .get("data", {})
        .get("content", {})
        .get("fields", {})
        or {}
    )


def _normalize_object_id_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        out: list[str] = []
        for item in raw:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                oid = item.get("id") or item.get("bytes")
                if oid:
                    out.append(str(oid))
        return out
    return []


def resolve_beneficiary_vault_id(
    profile: NetworkProfile,
    beneficiary_address: str,
) -> str | None:
    """GraphQL first, then None (callers may scan RPC separately)."""
    query = """
    query VaultForBeneficiary($addr: MySoAddress!) {
      pocBeneficiaryVaultByBeneficiary(beneficiary: $addr) { vaultId }
    }
    """
    try:
        data = _graphql_post(profile, query, {"addr": beneficiary_address})
        vault_id = (data.get("pocBeneficiaryVaultByBeneficiary") or {}).get("vaultId")
        if vault_id:
            return str(vault_id)
    except Exception as exc:
        logger.warning("GraphQL vault lookup failed", beneficiary=beneficiary_address, error=str(exc))
    return None


def resolve_username_beneficiary_shard_id(
    profile: NetworkProfile,
    username: str,
    *,
    directory_id: str | None = None,
) -> str:
    directory = directory_id or (profile.objects or {}).get("username_beneficiary_directory")
    if not directory:
        directory = __import__("os").environ.get("MYSO_POC_USERNAME_BENEFICIARY_DIRECTORY_ID")
    if not directory:
        raise RuntimeError("username beneficiary directory id not configured")

    fields = _rpc_get_object_fields(profile.rpc_url, str(directory))
    shard_ids = _normalize_object_id_list(fields.get("shard_ids"))
    if not shard_ids:
        raise RuntimeError(f"No shard_ids on directory {directory}")

    idx = shard_index_for_username(username)
    if idx >= len(shard_ids):
        raise RuntimeError(f"Shard index {idx} out of range for directory (len={len(shard_ids)})")
    return shard_ids[idx]


def fetch_username_beneficiary_fields(
    profile: NetworkProfile,
    beneficiary_id: str,
) -> dict[str, Any]:
    fields = _rpc_get_object_fields(profile.rpc_url, beneficiary_id)
    verification = fields.get("verification") or {}
    creator_identity = fields.get("creator_identity") or {}
    return {
        "status": normalize_beneficiary_status(fields.get("status")),
        "username": fields.get("username"),
        "required_x_handle": verification.get("required_x_handle") if isinstance(verification, dict) else None,
        "creator_identity_source": creator_identity.get("source") if isinstance(creator_identity, dict) else None,
        "creator_identity_hash": creator_identity.get("identity_hash") if isinstance(creator_identity, dict) else None,
        "vault_id": fields.get("vault_id"),
        "beneficiary_address": fields.get("beneficiary_address"),
        "raw": fields,
    }


def fetch_username_beneficiary_by_username(
    profile: NetworkProfile,
    username: str,
) -> dict[str, Any] | None:
    query = """
    query BeneficiaryByUsername($username: String!) {
      pocUsernameBeneficiaryByUsername(username: $username) {
        beneficiaryId
        beneficiaryAddress
        username
        creatorIdentitySource
        creatorIdentityHash
      }
    }
    """
    try:
        data = _graphql_post(profile, query, {"username": username})
    except Exception as exc:
        logger.warning("username beneficiary lookup failed", username=username, error=str(exc))
        return None
    row = data.get("pocUsernameBeneficiaryByUsername")
    return row if isinstance(row, dict) else None


def fetch_username_beneficiary_by_identity(
    profile: NetworkProfile,
    identity_hash: str,
    *,
    identity_source: int = 1,
) -> dict[str, Any] | None:
    from app.chain.move_address import parse_identity_hash

    ih_bytes = parse_identity_hash(identity_hash)
    query = """
    query BeneficiaryByIdentity($source: Int!, $hash: String!) {
      pocUsernameBeneficiaryByIdentity(identitySource: $source, identityHash: $hash) {
        beneficiaryId
        beneficiaryAddress
        username
        creatorIdentitySource
        creatorIdentityHash
      }
    }
    """
    try:
        data = _graphql_post(
            profile,
            query,
            {"source": int(identity_source), "hash": "0x" + ih_bytes.hex()},
        )
    except Exception as exc:
        logger.warning(
            "identity beneficiary lookup failed",
            identity_hash=identity_hash,
            error=str(exc),
        )
        return None
    row = data.get("pocUsernameBeneficiaryByIdentity")
    return row if isinstance(row, dict) else None


def fetch_username_availability(profile: NetworkProfile, username: str) -> bool:
    query = """
    query UsernameAvailable($username: String!) {
      usernameAvailability(username: $username) { available }
    }
    """
    data = _graphql_post(profile, query, {"username": username})
    row = data.get("usernameAvailability") or {}
    return bool(row.get("available"))


def fetch_profile_for_username(
    profile: NetworkProfile,
    username: str,
) -> dict[str, str] | None:
    """On-chain profile object id and owner when the username is registered and has no vault."""
    query = """
    query ProfileUsername($username: String!) {
      usernameRegistryEntry(username: $username) { profileId username }
    }
    """
    try:
        data = _graphql_post(profile, query, {"username": username})
    except Exception as exc:
        logger.warning("username registry lookup failed", username=username, error=str(exc))
        return None
    entry = data.get("usernameRegistryEntry")
    if not isinstance(entry, dict) or not entry.get("profileId"):
        return None
    profile_id = str(entry["profileId"])
    fields = _rpc_get_object_fields(profile.rpc_url, profile_id)
    owner = fields.get("owner")
    if not owner:
        return None
    return {"profile_id": profile_id, "owner": str(owner), "username": username}


def fetch_vault_beneficiary(profile: NetworkProfile, vault_id: str) -> str | None:
    query = """
    query VaultById($id: ID!) {
      pocBeneficiaryVaultByVaultId(vaultId: $id) { vaultId beneficiary }
    }
    """
    try:
        data = _graphql_post(profile, query, {"id": vault_id})
    except Exception as exc:
        logger.warning("vault lookup failed", vault_id=vault_id, error=str(exc))
        return None
    vault = data.get("pocBeneficiaryVaultByVaultId") or {}
    beneficiary = vault.get("beneficiary")
    return str(beneficiary) if beneficiary else None


def resolve_beneficiary_object_for_identity(
    profile: NetworkProfile,
    identity_hash: str,
    *,
    directory_id: str | None = None,
    identity_source: int = 1,
) -> str | None:
    """Lookup beneficiary object id from the indexer identity index."""
    _ = directory_id
    row = fetch_username_beneficiary_by_identity(
        profile,
        identity_hash,
        identity_source=identity_source,
    )
    if not row:
        return None
    beneficiary_id = row.get("beneficiaryId")
    return str(beneficiary_id) if beneficiary_id else None


def get_network_profile(network: str) -> NetworkProfile:
    return load_network_profile(network)
