"""Network-aware chain transaction submitter wrapping MySocialClient."""

from __future__ import annotations

import hashlib
import os
from typing import Any

import structlog

from app.network_config import bootstrap_network_session, chain_writes_allowed, load_network_profile
from app.services.myso_client import MySocialClient, init_myso_client
from app.services.myso_wallet import MySocialWallet

logger = structlog.get_logger()


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
        shard_index: int = 0,
    ) -> dict:
        if not self.writes_enabled:
            return self._mock_tx(
                "create_username_beneficiary",
                {"username": username, "identity_hash": identity_hash},
            )
        return self._submit_admin_move_call(
            "create_username_beneficiary",
            [username, identity_hash, shard_index],
        )

    def claim_username_beneficiary(
        self,
        *,
        identity_hash: str,
        claimant_address: str,
    ) -> dict:
        if not self.writes_enabled:
            return self._mock_tx(
                "claim_username_beneficiary",
                {"identity_hash": identity_hash, "claimant": claimant_address},
            )
        if not self.client:
            raise RuntimeError("MySocial client unavailable for chain writes")
        return self._submit_oracle_move_call(
            "claim_username_beneficiary",
            [identity_hash, claimant_address],
        )

    def _submit_oracle_move_call(self, function: str, args: list) -> dict:
        if not self.client:
            raise RuntimeError("MySocial client unavailable")
        objs = self.profile.objects or {}
        data = {
            "packageObjectId": self.client.package_id,
            "module": "proof_of_creativity",
            "function": function,
            "typeArguments": [],
            "arguments": [
                objs.get("poc_config") or self.client.config_id,
                *args,
            ],
        }
        return self._execute_move_call(data)

    def _submit_admin_move_call(self, function: str, args: list) -> dict:
        admin_key = os.getenv("POC_ADMIN_PRIVATE_KEY") or os.getenv(
            self.profile.signers.admin_private_key_env, ""
        )
        if not admin_key:
            raise RuntimeError("Admin private key not configured for username beneficiary provisioning")
        wallet = MySocialWallet(private_key=admin_key)
        objs = self.profile.objects or {}
        package = objs.get("poc_package") or os.getenv("MYSO_POC_PACKAGE_ID")
        data = {
            "packageObjectId": package,
            "module": "proof_of_creativity",
            "function": function,
            "typeArguments": [],
            "arguments": [
                objs.get("poc_config") or os.getenv("MYSO_POC_CONFIG_ID"),
                objs.get("username_beneficiary_directory"),
                objs.get("username_registry"),
                *args,
            ],
        }
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
        return client._submit_move_call(data, wallet=wallet or client.wallet)

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
