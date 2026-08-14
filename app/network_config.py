"""Network profile loading and oracle runtime settings."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config" / "networks"

SUPPORTED_NETWORKS = ("localnet", "testnet", "mainnet")


class StorageConfig(BaseModel):
    use_walrus: bool = False
    use_gcs: bool = False
    use_local: bool = True
    local_path: str = "./data/blobs"
    walrus_endpoint: str = "https://api.walrus.xyz"
    gcs_bucket_name: str | None = None


class SignerConfig(BaseModel):
    oracle_private_key_env: str = "ORACLE_PRIVATE_KEY"
    admin_private_key_env: str = "POC_ADMIN_PRIVATE_KEY"


class GrpcSyncConfig(BaseModel):
    mock_mode: bool = False
    fixture_path: str | None = None
    start_checkpoint: int | None = None
    sync_mode: str = "checkpoint_v2"
    event_stream_id: str = ""
    batch_size: int = 1000
    poll_interval_seconds: int = 10
    max_pagination_iterations: int = 50
    checkpoint_catchup_batch_size: int = 10


class NetworkProfile(BaseModel):
    network: str
    rpc_url: str
    grpc_url: str
    grpc_tls: bool = True
    faucet_url: str | None = None
    chain_writes_enabled: bool = True
    graphql_url: str = "http://127.0.0.1:9125/graphql"
    platform_package_address: str = "0x50c1"
    objects: dict[str, Any] = Field(default_factory=dict)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    signers: SignerConfig = Field(default_factory=SignerConfig)
    grpc_sync: GrpcSyncConfig = Field(default_factory=GrpcSyncConfig)


class Settings(BaseModel):
    myso_network: str = "localnet"
    myso_networks: list[str] = Field(default_factory=list)
    database_url: str = ""
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    ws_path: str = "/ws"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    ws_auth_required: bool = False
    oracle_worker_concurrency: int = 4
    oracle_max_retries: int = 5
    off_network_force_escrow: bool = True
    mainnet_writes_enabled: bool = False


def active_networks() -> list[str]:
    raw = os.getenv("MYSO_NETWORKS", "").strip()
    if raw:
        return [n.strip() for n in raw.split(",") if n.strip()]
    return [os.getenv("MYSO_NETWORK", "localnet").strip() or "localnet"]


@lru_cache(maxsize=8)
def _load_network_profile_from_yaml(network: str) -> NetworkProfile:
    if network not in SUPPORTED_NETWORKS:
        raise ValueError(f"Unsupported network: {network}")
    path = CONFIG_DIR / f"{network}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Network profile not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return NetworkProfile(**data)


def _apply_env_overrides(profile: NetworkProfile) -> NetworkProfile:
    """Let .env / shell overrides win over Docker-oriented yaml defaults."""
    updates: dict[str, Any] = {}
    rpc = os.getenv("MYSOCIAL_RPC_URL", "").strip()
    if rpc:
        updates["rpc_url"] = rpc
        grpc = os.getenv("MYSO_GRPC_URL", "").strip() or rpc
        updates["grpc_url"] = grpc
    graphql = os.getenv("GRAPHQL_URL", "").strip()
    if graphql:
        updates["graphql_url"] = graphql

    grpc_tls = os.getenv("MYSO_GRPC_TLS", "").strip().lower()
    if grpc_tls in ("0", "false", "no"):
        updates["grpc_tls"] = False
    elif grpc_tls in ("1", "true", "yes"):
        updates["grpc_tls"] = True
    else:
        grpc_url = str(updates.get("grpc_url", profile.grpc_url))
        if grpc_url.startswith("http://"):
            updates["grpc_tls"] = False

    if updates:
        return profile.model_copy(update=updates)
    return profile


def load_network_profile(network: str) -> NetworkProfile:
    return _apply_env_overrides(_load_network_profile_from_yaml(network))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    cors = os.getenv("CORS_ORIGINS", "http://localhost:3000")
    return Settings(
        myso_network=os.getenv("MYSO_NETWORK", "localnet"),
        myso_networks=active_networks(),
        database_url=os.getenv("DATABASE_URL") or os.getenv("DB_DSN") or "",
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=int(os.getenv("API_PORT", os.getenv("PORT", "8000"))),
        ws_path=os.getenv("WS_PATH", "/ws"),
        cors_origins=[o.strip() for o in cors.split(",") if o.strip()],
        ws_auth_required=os.getenv("WS_AUTH_REQUIRED", "false").lower() == "true",
        oracle_worker_concurrency=int(os.getenv("ORACLE_WORKER_CONCURRENCY", "4")),
        oracle_max_retries=int(os.getenv("ORACLE_MAX_RETRIES", "5")),
        off_network_force_escrow=os.getenv("OFF_NETWORK_FORCE_ESCROW", "true").lower() == "true",
        mainnet_writes_enabled=os.getenv("MAINNET_WRITES_ENABLED", "false").lower() == "true",
    )


def chain_writes_allowed(profile: NetworkProfile) -> bool:
    if profile.network == "mainnet":
        return get_settings().mainnet_writes_enabled and profile.chain_writes_enabled
    return profile.chain_writes_enabled


def apply_profile_to_env(profile: NetworkProfile) -> None:
    """Inject network profile object IDs into process env for MySocialClient."""
    if not os.getenv("MYSOCIAL_RPC_URL", "").strip():
        os.environ["MYSOCIAL_RPC_URL"] = profile.rpc_url
    objs = profile.objects or {}
    mapping = {
        "MYSO_POC_PACKAGE_ID": objs.get("poc_package") or objs.get("package"),
        "MYSO_POC_CONFIG_ID": objs.get("poc_config"),
        "MYSO_POC_REGISTRY_ID": objs.get("poc_registry"),
        "MYSO_POC_VAULT_DIRECTORY_ID": objs.get("poc_vault_directory"),
        "MYSO_TOKEN_REGISTRY_ID": objs.get("token_registry"),
        "MYSO_USERNAME_REGISTRY_ID": objs.get("username_registry"),
        "MYSO_POC_USERNAME_BENEFICIARY_DIRECTORY_ID": objs.get("username_beneficiary_directory"),
        "MYSO_PROFILE_CONFIG_ID": objs.get("profile_config"),
        "MYSO_MEMORY_REGISTRY_ID": objs.get("memory_registry"),
        "MYSO_AI_CREDIT_CONFIG_ID": objs.get("ai_credit_config"),
        "MYSO_POC_BENEFICIARY_ADMIN_CAP_ID": objs.get("poc_beneficiary_admin_cap"),
        "MYSO_CLOCK_OBJECT_ID": objs.get("clock"),
    }
    for key, val in mapping.items():
        if not os.getenv(key) and val:
            os.environ[key] = str(val)

    oracle_env = profile.signers.oracle_private_key_env
    admin_env = profile.signers.admin_private_key_env
    if os.getenv(oracle_env):
        os.environ["MYSO_ORACLE_PRIVATE_KEY"] = os.environ[oracle_env]
    if os.getenv(admin_env):
        os.environ["POC_ADMIN_PRIVATE_KEY"] = os.environ[admin_env]


def bootstrap_network_session(network: str) -> dict[str, str]:
    """
    Refresh PoC object IDs from GraphQL (when enabled), then apply network profile to env.

    GraphQL/env values take precedence over static entries in config/networks/*.yaml.
    """
    profile = load_network_profile(network)
    resolved: dict[str, str] = {}
    try:
        from app.chain.graphql_session import refresh_poc_session_objects

        resolved = refresh_poc_session_objects(profile)
        for key, value in resolved.items():
            if value:
                os.environ[key] = value
    except Exception as exc:
        import structlog

        structlog.get_logger().warning(
            "PoC session refresh failed — using .env / network profile fallbacks",
            network=network,
            error=str(exc),
        )
    apply_profile_to_env(profile)
    return resolved


def validate_registry_objects(profile: NetworkProfile) -> list[str]:
    """Return missing required registry object keys for startup validation."""
    required = (
        "poc_registry",
        "poc_config",
        "username_registry",
        "poc_vault_directory",
        "username_beneficiary_directory",
        "poc_beneficiary_admin_cap",
    )
    env_by_key = {
        "poc_registry": "MYSO_POC_REGISTRY_ID",
        "poc_config": "MYSO_POC_CONFIG_ID",
        "username_registry": "MYSO_USERNAME_REGISTRY_ID",
        "poc_vault_directory": "MYSO_POC_VAULT_DIRECTORY_ID",
        "username_beneficiary_directory": "MYSO_POC_USERNAME_BENEFICIARY_DIRECTORY_ID",
        "poc_beneficiary_admin_cap": "MYSO_POC_BENEFICIARY_ADMIN_CAP_ID",
    }
    objs = profile.objects or {}
    missing = [
        key
        for key in required
        if not objs.get(key) and not os.getenv(env_by_key[key], "").strip()
    ]
    return missing


def bootstrap_active_network_sessions() -> None:
    """Bootstrap all active networks (MYSO_NETWORK or MYSO_NETWORKS)."""
    for network in active_networks():
        bootstrap_network_session(network)
