"""Refresh PoC oracle session object IDs from the MySocial GraphQL indexer."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import requests
import structlog

from app.network_config import NetworkProfile

logger = structlog.get_logger()

# GraphQL alias -> (env var, profile.objects key)
POC_OBJECT_ALIASES: dict[str, tuple[str, str]] = {
    "pocConfig": ("MYSO_POC_CONFIG_ID", "poc_config"),
    "pocRegistry": ("MYSO_POC_REGISTRY_ID", "poc_registry"),
    "pocVaultDirectory": ("MYSO_POC_VAULT_DIRECTORY_ID", "poc_vault_directory"),
    "tokenRegistry": ("MYSO_TOKEN_REGISTRY_ID", "token_registry"),
    "usernameRegistry": ("MYSO_USERNAME_REGISTRY_ID", "username_registry"),
    "pocUsernameBeneficiaryDirectory": (
        "MYSO_POC_USERNAME_BENEFICIARY_DIRECTORY_ID",
        "username_beneficiary_directory",
    ),
    "profileConfig": ("MYSO_PROFILE_CONFIG_ID", "profile_config"),
    "memoryRegistry": ("MYSO_MEMORY_REGISTRY_ID", "memory_registry"),
    "aiCreditConfig": ("MYSO_AI_CREDIT_CONFIG_ID", "ai_credit_config"),
    "pocBeneficiaryAdminCap": ("MYSO_POC_BENEFICIARY_ADMIN_CAP_ID", "poc_beneficiary_admin_cap"),
}

_MOVE_TYPE_PACKAGE_RE = re.compile(r"^(0x[0-9a-fA-F]+)::")


def session_refresh_enabled() -> bool:
    raw = os.getenv("MYSO_REFRESH_SESSION_OBJECTS", "true").strip().lower()
    return raw not in ("0", "false", "no", "off")


def resolve_graphql_url(profile: NetworkProfile) -> str:
    return (
        os.getenv("GRAPHQL_URL", "").strip()
        or profile.graphql_url.strip()
        or "http://127.0.0.1:9125/graphql"
    )


def resolve_platform_package_address(profile: NetworkProfile) -> str:
    return (
        os.getenv("MYSO_PLATFORM_PACKAGE_ADDRESS", "").strip()
        or profile.platform_package_address.strip()
        or "0x50c1"
    )


def build_poc_session_query(platform_package: str) -> str:
    """Build GraphQL query for shared PoC / platform objects (localnet E2E pattern)."""
    pkg = platform_package
    shared_filters = [
        ("pocConfig", f"{pkg}::proof_of_creativity::PoCConfig"),
        ("pocRegistry", f"{pkg}::proof_of_creativity::PoCRegistry"),
        ("pocVaultDirectory", f"{pkg}::poc_vault::PoCVaultDirectory"),
        ("tokenRegistry", f"{pkg}::social_proof_tokens::TokenRegistry"),
        ("usernameRegistry", f"{pkg}::profile::UsernameRegistry"),
        ("pocUsernameBeneficiaryDirectory", f"{pkg}::poc_username_beneficiary::PoCUsernameBeneficiaryDirectory"),
        ("profileConfig", f"{pkg}::profile::ProfileConfig"),
        ("memoryRegistry", f"{pkg}::memory::MemoryRegistry"),
        ("aiCreditConfig", f"{pkg}::ai_credit::AiCreditConfig"),
    ]
    owned_filters = [
        ("pocBeneficiaryAdminCap", f"{pkg}::poc_username_beneficiary::PoCBeneficiaryAdminCap"),
    ]
    parts = []
    for alias, move_type in shared_filters:
        parts.append(
            f"{alias}: objects("
            f'filter: {{ type: "{move_type}", ownerKind: SHARED }}, first: 1'
            f") {{ nodes {{ address }} }}"
        )
    for alias, move_type in owned_filters:
        parts.append(
            f"{alias}: objects("
            f'filter: {{ type: "{move_type}" }}, last: 1'
            f") {{ nodes {{ address }} }}"
        )
    return f"query PoCOracleSessionObjects {{ {' '.join(parts)} }}"


def _graphql_post(url: str, query: str, timeout: float = 15.0) -> dict[str, Any]:
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        data=json.dumps({"query": query}),
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise RuntimeError(f"GraphQL errors: {payload['errors']}")
    return payload.get("data") or {}


def _extract_address(data: dict[str, Any], alias: str) -> str | None:
    nodes = (data.get(alias) or {}).get("nodes") or []
    if not nodes:
        return None
    address = nodes[0].get("address")
    if isinstance(address, str) and address.strip():
        return address.strip()
    return None


def extract_package_address_from_move_type(move_type: str) -> str | None:
    match = _MOVE_TYPE_PACKAGE_RE.match(move_type.strip())
    if match:
        return match.group(1)
    return None


def resolve_package_id_via_rpc(rpc_url: str, object_id: str, timeout: float = 10.0) -> str | None:
    """Derive published package address from an on-chain object's Move type."""
    response = requests.post(
        rpc_url,
        headers={"Content-Type": "application/json"},
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "myso_getObject",
            "params": [object_id, {"showType": True}],
        },
        timeout=timeout,
    )
    response.raise_for_status()
    result = response.json()
    if result.get("error"):
        logger.warning("RPC getObject failed during package resolve", object_id=object_id, error=result["error"])
        return None
    move_type = result.get("result", {}).get("data", {}).get("type")
    if not isinstance(move_type, str):
        return None
    package = extract_package_address_from_move_type(move_type)
    if package:
        logger.info("Resolved PoC package from object type", object_id=object_id, package_id=package, move_type=move_type)
    return package


def refresh_poc_session_objects(profile: NetworkProfile) -> dict[str, str]:
    """
    Query GraphQL for current shared PoC object addresses and export them to os.environ.

    Returns mapping of env var name -> resolved address.
    """
    if not session_refresh_enabled():
        logger.info("PoC session refresh disabled", network=profile.network)
        return {}

    graphql_url = resolve_graphql_url(profile)
    platform_package = resolve_platform_package_address(profile)
    query = build_poc_session_query(platform_package)

    logger.info(
        "Refreshing PoC session objects from GraphQL",
        network=profile.network,
        graphql_url=graphql_url,
        platform_package=platform_package,
    )

    data = _graphql_post(graphql_url, query)
    resolved: dict[str, str] = {}

    for alias, (env_key, _profile_key) in POC_OBJECT_ALIASES.items():
        address = _extract_address(data, alias)
        if address:
            os.environ[env_key] = address
            resolved[env_key] = address
            logger.info("PoC session object resolved", alias=alias, env_key=env_key, address=address)

    config_id = resolved.get("MYSO_POC_CONFIG_ID") or os.getenv("MYSO_POC_CONFIG_ID")
    package_id: str | None = None
    if config_id:
        package_id = resolve_package_id_via_rpc(profile.rpc_url, config_id)
    if not package_id:
        package_id = os.getenv("MYSO_POC_PACKAGE_ID") or platform_package
    if package_id:
        os.environ["MYSO_POC_PACKAGE_ID"] = package_id
        resolved["MYSO_POC_PACKAGE_ID"] = package_id

    missing = [alias for alias in POC_OBJECT_ALIASES if POC_OBJECT_ALIASES[alias][0] not in resolved]
    if missing:
        logger.warning(
            "PoC session refresh incomplete — some objects not found in GraphQL",
            network=profile.network,
            missing_aliases=missing,
        )
    else:
        logger.info("PoC session refresh complete", network=profile.network, object_count=len(resolved))

    return resolved
