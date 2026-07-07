"""Network-aware chain transaction submitter wrapping MySocialClient."""

from __future__ import annotations

import hashlib
import os
from typing import Any

import structlog

from app.chain.move_calls import (
    build_claim_username_beneficiary_call,
    build_create_username_beneficiary_call,
)
from app.network_config import NetworkProfile, bootstrap_network_session, chain_writes_allowed, load_network_profile
from app.services.myso_client import MySocialClient, init_myso_client
from app.services.myso_wallet import MySocialWallet
from app.services.poc_chain_helpers import (
    resolve_beneficiary_object_for_identity,
    resolve_username_beneficiary_shard_id,
)

logger = structlog.get_logger()


def _env_object_id(profile: NetworkProfile, env_key: str, profile_key: str) -> str | None:
    val = os.getenv(env_key, "").strip()
    if val:
        return val
    objs = profile.objects or {}
    raw = objs.get(profile_key)
    if raw:
        return str(raw).strip()
    return None


def _require_object_id(profile: NetworkProfile, env_key: str, profile_key: str, label: str) -> str:
    oid = _env_object_id(profile, env_key, profile_key)
    if not oid:
        raise RuntimeError(f"Missing on-chain object id for {label} ({env_key})")
    return oid


class TransactionSubmitter:
    def __init__(self, network: str) -> None:
        self.network = network
        self.profile = load_network_profile(network)
        bootstrap_network_session(network)
        os.environ["MYSO_INTEGRATION_ENABLED"] = "true"
        self.client: MySocialClient | None = init_myso_client()

    @property
    def writes_enabled(self) -> bool:
        return chain_writes_allowed(self.profile)

    def get_poc_config(self) -> dict:
        if self.client:
            cfg = self.client.get_poc_config()
            return cfg
        from app.services.poc_utils import OFFCHAIN_DEFAULT_POC_CONFIG

        return dict(OFFCHAIN_DEFAULT_POC_CONFIG)

    def submit_analysis(self, **kwargs: Any) -> dict:
        if not self.writes_enabled:
            return self._mock_tx("analyze_and_update_post", kwargs)
        if not self.client:
            raise RuntimeError("MySocial client unavailable for chain writes")
        return self.client.submit_poc_analysis(**kwargs)

    def create_username_beneficiary(
        self,
        *,
        username: str,
        identity_hash: str,
        required_x_handle: str | None = None,
    ) -> dict:
        if not self.writes_enabled:
            return self._mock_tx(
                "create_username_beneficiary",
                {"username": username, "identity_hash": identity_hash},
            )
        package_id = _env_object_id(self.profile, "MYSO_POC_PACKAGE_ID", "poc_package") or (
            self.client.package_id if self.client else None
        )
        if not package_id:
            raise RuntimeError("PoC package id not configured")

        directory_id = _require_object_id(
            self.profile,
            "MYSO_POC_USERNAME_BENEFICIARY_DIRECTORY_ID",
            "username_beneficiary_directory",
            "PoCUsernameBeneficiaryDirectory",
        )
        shard_id = resolve_username_beneficiary_shard_id(
            self.profile,
            username,
            directory_id=directory_id,
        )

        data = build_create_username_beneficiary_call(
            package_id=str(package_id),
            admin_cap_id=_require_object_id(
                self.profile,
                "MYSO_POC_BENEFICIARY_ADMIN_CAP_ID",
                "poc_beneficiary_admin_cap",
                "PoCBeneficiaryAdminCap",
            ),
            directory_id=directory_id,
            shard_id=shard_id,
            vault_directory_id=_require_object_id(
                self.profile,
                "MYSO_POC_VAULT_DIRECTORY_ID",
                "poc_vault_directory",
                "PoCVaultDirectory",
            ),
            username_registry_id=_require_object_id(
                self.profile,
                "MYSO_USERNAME_REGISTRY_ID",
                "username_registry",
                "UsernameRegistry",
            ),
            username=username,
            identity_hash=identity_hash,
            required_x_handle=required_x_handle or username,
            clock_id=_require_object_id(self.profile, "MYSO_CLOCK_OBJECT_ID", "clock", "Clock"),
        )
        return self._submit_admin_move_call(data)

    def claim_username_beneficiary(
        self,
        *,
        identity_hash: str,
        claimant_address: str,
        beneficiary_id: str | None = None,
        evidence_hash: bytes | None = None,
        attested_x_handle: str | None = None,
        display_name: str = "",
        bio: str = "",
        profile_picture_url: str = "",
        cover_photo_url: str = "",
    ) -> dict:
        if not self.writes_enabled:
            return self._mock_tx(
                "claim_username_beneficiary",
                {"identity_hash": identity_hash, "claimant": claimant_address},
            )
        if not self.client:
            raise RuntimeError("MySocial client unavailable for chain writes")

        package_id = self.client.package_id or _env_object_id(
            self.profile, "MYSO_POC_PACKAGE_ID", "poc_package"
        )
        if not package_id:
            raise RuntimeError("PoC package id not configured")

        beneficiary = beneficiary_id or resolve_beneficiary_object_for_identity(
            self.profile, identity_hash
        )
        if not beneficiary:
            raise RuntimeError(f"Could not resolve PoCUsernameBeneficiary for identity {identity_hash}")

        if evidence_hash is None:
            evidence_hash = b""
        if not attested_x_handle:
            attested_x_handle = identity_hash[:16] if len(identity_hash) > 16 else identity_hash

        directory_id = _require_object_id(
            self.profile,
            "MYSO_POC_USERNAME_BENEFICIARY_DIRECTORY_ID",
            "username_beneficiary_directory",
            "PoCUsernameBeneficiaryDirectory",
        )
        from app.services.poc_chain_helpers import fetch_username_beneficiary_fields

        fields = fetch_username_beneficiary_fields(self.profile, beneficiary)
        username = str(fields.get("username") or attested_x_handle)
        shard_id = resolve_username_beneficiary_shard_id(
            self.profile,
            username,
            directory_id=directory_id,
        )

        data = build_claim_username_beneficiary_call(
            package_id=str(package_id),
            poc_config_id=_require_object_id(
                self.profile, "MYSO_POC_CONFIG_ID", "poc_config", "PoCConfig"
            ),
            profile_config_id=_require_object_id(
                self.profile, "MYSO_PROFILE_CONFIG_ID", "profile_config", "ProfileConfig"
            ),
            directory_id=directory_id,
            shard_id=shard_id,
            username_registry_id=_require_object_id(
                self.profile, "MYSO_USERNAME_REGISTRY_ID", "username_registry", "UsernameRegistry"
            ),
            memory_registry_id=_require_object_id(
                self.profile, "MYSO_MEMORY_REGISTRY_ID", "memory_registry", "MemoryRegistry"
            ),
            ai_credit_config_id=_require_object_id(
                self.profile, "MYSO_AI_CREDIT_CONFIG_ID", "ai_credit_config", "AiCreditConfig"
            ),
            beneficiary_id=beneficiary,
            evidence_hash=evidence_hash,
            attested_x_handle=attested_x_handle,
            display_name=display_name,
            bio=bio,
            profile_picture_url=profile_picture_url,
            cover_photo_url=cover_photo_url,
            wallet=claimant_address,
            clock_id=_require_object_id(self.profile, "MYSO_CLOCK_OBJECT_ID", "clock", "Clock"),
        )
        return self._execute_move_call(data)

    def _submit_admin_move_call(self, data: dict) -> dict:
        admin_key = os.getenv("POC_ADMIN_PRIVATE_KEY") or os.getenv(
            self.profile.signers.admin_private_key_env, ""
        )
        if not admin_key:
            raise RuntimeError("Admin private key not configured for username beneficiary provisioning")
        wallet = MySocialWallet(private_key=admin_key)
        prev = self.client
        self.client = MySocialClient(wallet=wallet)
        try:
            return self._execute_move_call(data, wallet=wallet)
        finally:
            self.client = prev

    def _execute_move_call(
        self,
        data: dict,
        *,
        wallet: MySocialWallet | None = None,
        signature: str | None = None,
        tx_data: dict | None = None,
    ) -> dict:
        client = self.client
        if not client:
            raise RuntimeError("MySocial client unavailable")
        result = client._submit_move_call(data, wallet=wallet or client.wallet)
        result["move_function"] = data.get("function")
        return result

    def _mock_tx(self, function: str, payload: dict) -> dict:
        digest_src = f"{self.network}:{function}:{payload!r}"
        digest = "0x" + hashlib.sha256(digest_src.encode()).hexdigest()[:64]
        logger.info("Mock chain tx (writes disabled)", network=self.network, function=function, digest=digest)
        return {
            "success": True,
            "tx_hash": digest,
            "status": {"status": "success"},
            "events": [],
            "mock": True,
            "move_function": function,
        }
