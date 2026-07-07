"""Off-network username beneficiary vault provisioning and claims."""

from __future__ import annotations

import structlog

from app.chain.move_address import derive_beneficiary_address
from app.chain.rpc_client import TransactionSubmitter
from app.db.oracle_repository import UsernameBeneficiaryRepository
from app.services.events import event_bus

logger = structlog.get_logger()


class UsernameBeneficiaryService:
    def __init__(self, network: str) -> None:
        self.network = network
        self.repo = UsernameBeneficiaryRepository()
        self.submitter = TransactionSubmitter(network)

    def get_or_compute(self, identity_hash: str, *, username: str | None = None) -> dict:
        existing = self.repo.get(self.network, identity_hash)
        if existing and existing.get("beneficiary_address"):
            return existing
        address = derive_beneficiary_address(identity_hash)
        self.repo.upsert(
            self.network,
            identity_hash,
            username=username,
            beneficiary_address=address,
        )
        return self.repo.get(self.network, identity_hash) or {"beneficiary_address": address}

    async def ensure_provisioned(
        self,
        *,
        post_id: str,
        identity_hash: str,
        username: str | None = None,
    ) -> str:
        record = self.get_or_compute(identity_hash, username=username)
        address = record.get("beneficiary_address") or derive_beneficiary_address(identity_hash)
        if record.get("provision_tx_digest"):
            return str(address)

        username_value = username or identity_hash[:16]
        result = self.submitter.create_username_beneficiary(
            username=username_value,
            identity_hash=identity_hash,
        )
        tx = result.get("tx_hash")
        self.repo.upsert(
            self.network,
            identity_hash,
            username=username_value,
            beneficiary_address=address,
            provision_tx_digest=tx,
        )
        await event_bus.publish(
            "post.beneficiary.provisioned",
            {
                "network": self.network,
                "post_id": post_id,
                "beneficiary_address": address,
                "vault_id": record.get("vault_object_id"),
            },
        )
        return str(address)

    async def claim(
        self,
        identity_hash: str,
        claimant_address: str,
        *,
        beneficiary_id: str | None = None,
        evidence_hash: bytes | None = None,
        attested_x_handle: str | None = None,
        display_name: str = "",
        bio: str = "",
        profile_picture_url: str = "",
        cover_photo_url: str = "",
    ) -> dict:
        result = self.submitter.claim_username_beneficiary(
            identity_hash=identity_hash,
            claimant_address=claimant_address,
            beneficiary_id=beneficiary_id,
            evidence_hash=evidence_hash,
            attested_x_handle=attested_x_handle,
            display_name=display_name,
            bio=bio,
            profile_picture_url=profile_picture_url,
            cover_photo_url=cover_photo_url,
        )
        tx = result.get("tx_hash")
        if tx:
            self.repo.mark_claimed(self.network, identity_hash, str(tx))
        await event_bus.publish(
            "beneficiary.claimed",
            {
                "network": self.network,
                "identity_hash": identity_hash,
                "claimant_address": claimant_address,
                "tx_digest": tx,
            },
        )
        return result
