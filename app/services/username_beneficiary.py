"""Off-network username beneficiary vault provisioning and claims."""

from __future__ import annotations

import hashlib

import structlog

from app.chain.rpc_client import TransactionSubmitter
from app.db.oracle_repository import UsernameBeneficiaryRepository
from app.services.events import event_bus

logger = structlog.get_logger()


def derive_beneficiary_address(identity_hash: str, network: str) -> str:
    digest = hashlib.sha256(f"{network}:beneficiary:{identity_hash}".encode()).hexdigest()
    return "0x" + digest[:64]


class UsernameBeneficiaryService:
    def __init__(self, network: str) -> None:
        self.network = network
        self.repo = UsernameBeneficiaryRepository()
        self.submitter = TransactionSubmitter(network)

    def get_or_compute(self, identity_hash: str, *, username: str | None = None) -> dict:
        existing = self.repo.get(self.network, identity_hash)
        if existing and existing.get("beneficiary_address"):
            return existing
        address = derive_beneficiary_address(identity_hash, self.network)
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
        address = record.get("beneficiary_address") or derive_beneficiary_address(identity_hash, self.network)
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

    async def claim(self, identity_hash: str, claimant_address: str) -> dict:
        result = self.submitter.claim_username_beneficiary(
            identity_hash=identity_hash,
            claimant_address=claimant_address,
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
