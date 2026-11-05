"""Vault lifecycle orchestration for lazy provisioning and discovery lifecycle updates."""

from __future__ import annotations

import structlog

from app.discovery.store import DiscoveryStore
from app.network_config import load_network_profile
from app.services.events import event_bus
from app.services.poc_chain_helpers import (
    resolve_beneficiary_object_for_identity,
    resolve_beneficiary_vault_id,
)
from app.services.username_beneficiary import UsernameBeneficiaryService

logger = structlog.get_logger()


class VaultLifecycleService:
    def __init__(self, network: str) -> None:
        self.network = network
        self.profile = load_network_profile(network)
        self.beneficiaries = UsernameBeneficiaryService(network)
        self.discovery = DiscoveryStore()

    async def ensure_off_network_vault(
        self,
        *,
        post_id: str,
        identity_hash: str,
        username: str | None,
        discovery_asset_id: str | None,
    ) -> tuple[str | None, bool]:
        """Provision lazy vault if needed; return (beneficiary_address, provisioned_now)."""
        record = self.beneficiaries.get_or_compute(identity_hash, username=username)
        already = bool(record.get("provision_tx_digest"))
        address = await self.beneficiaries.ensure_provisioned(
            post_id=post_id,
            identity_hash=identity_hash,
            username=username,
        )
        vault_object_id = record.get("vault_object_id")
        if not vault_object_id:
            beneficiary_id = resolve_beneficiary_object_for_identity(self.profile, identity_hash)
            if beneficiary_id:
                vault_object_id = resolve_beneficiary_vault_id(self.profile, beneficiary_id)
                if vault_object_id:
                    from app.db.oracle_repository import UsernameBeneficiaryRepository

                    UsernameBeneficiaryRepository().upsert(
                        self.network,
                        identity_hash,
                        vault_object_id=str(vault_object_id),
                    )

        provisioned_now = not already
        if provisioned_now:
            await event_bus.publish(
                "post.beneficiary.provisioned",
                {
                    "network": self.network,
                    "post_id": post_id,
                    "beneficiary_address": address,
                    "vault_id": vault_object_id,
                },
            )
            if discovery_asset_id:
                self.discovery.transition_asset(discovery_asset_id, "vault_created")
        return address, provisioned_now

    async def mark_escrow_active(self, post_id: str, identity_hash: str) -> None:
        await event_bus.publish(
            "vault.lifecycle.escrow_active",
            {"network": self.network, "post_id": post_id, "identity_hash": identity_hash},
        )

    async def mark_claimable(self, identity_hash: str, discovery_asset_id: str | None) -> None:
        from app.db.oracle_repository import UsernameBeneficiaryRepository

        UsernameBeneficiaryRepository().upsert(
            self.network,
            identity_hash,
            metadata={"claim_status": "unclaimed"},
        )
        if discovery_asset_id:
            self.discovery.transition_asset(discovery_asset_id, "vault_claimable")

    async def mark_claimed(
        self,
        *,
        identity_hash: str,
        claimant_address: str,
        discovery_asset_id: str | None,
    ) -> None:
        from app.db.oracle_repository import UsernameBeneficiaryRepository

        UsernameBeneficiaryRepository().upsert(
            self.network,
            identity_hash,
            metadata={"claim_status": "claimed", "claimant_address": claimant_address},
        )
        if discovery_asset_id:
            self.discovery.transition_asset(discovery_asset_id, "vault_claimed")
