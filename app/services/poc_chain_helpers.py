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


def resolve_beneficiary_object_for_identity(
    profile: NetworkProfile,
    identity_hash: str,
    *,
    directory_id: str | None = None,
) -> str | None:
    """Lookup beneficiary shared object id from directory.beneficiary_by_identity (RPC scan fallback)."""
    from app.chain.move_address import parse_identity_hash

    ih_bytes = parse_identity_hash(identity_hash)
    directory = directory_id or (profile.objects or {}).get("username_beneficiary_directory")
    if not directory:
        return None

    fields = _rpc_get_object_fields(profile.rpc_url, str(directory))
    # Table contents are not exposed via simple getObject; use GraphQL when available.
    query = """
    query BeneficiaryByIdentity($hash: String!) {
      pocUsernameBeneficiaries(filter: { identityHash: $hash }, first: 1) {
        nodes { beneficiaryId address }
      }
    }
    """
    try:
        data = _graphql_post(profile, query, {"hash": ih_bytes.hex()})
        nodes = (data.get("pocUsernameBeneficiaries") or {}).get("nodes") or []
        if nodes:
            return str(nodes[0].get("beneficiaryId") or nodes[0].get("address"))
    except Exception:
        pass

    _ = fields  # reserved for future table walk
    return None


def get_network_profile(network: str) -> NetworkProfile:
    return load_network_profile(network)
