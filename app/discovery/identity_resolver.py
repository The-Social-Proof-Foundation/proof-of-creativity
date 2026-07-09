"""Resolve discovered creator identity across DB, chain, and MySocial."""

from __future__ import annotations

from dataclasses import dataclass

from app.chain.move_address import canonical_registry_username
from app.discovery.identity import identity_hash_from_x_handle, resolve_identity_hash
from app.db.oracle_repository import UsernameBeneficiaryRepository
from app.network_config import load_network_profile
from app.services.poc_chain_helpers import (
    fetch_username_beneficiary_fields,
    resolve_beneficiary_object_for_identity,
    resolve_beneficiary_vault_id,
)


@dataclass(frozen=True)
class ResolvedIdentity:
    identity_hash: str | None
    x_handle: str | None
    beneficiary_address: str | None
    vault_object_id: str | None
    beneficiary_id: str | None
    mysocial_creator_address: str | None
    confidence: float
    source: str
    creator_candidate_id: str | None = None


class IdentityResolver:
    def __init__(self, network: str) -> None:
        self.network = network
        self.profile = load_network_profile(network)
        self.beneficiaries = UsernameBeneficiaryRepository()

    def resolve(
        self,
        *,
        creator_x_handle: str | None = None,
        identity_hash: str | None = None,
        creator_candidate_id: str | None = None,
        creator_confidence: float = 0.0,
    ) -> ResolvedIdentity:
        ih = resolve_identity_hash(creator_x_handle, identity_hash)
        handle = canonical_registry_username(creator_x_handle) if creator_x_handle else None
        if handle and not ih:
            ih = identity_hash_from_x_handle(handle)

        record = self.beneficiaries.get(self.network, ih) if ih else None
        beneficiary_id = None
        vault_object_id = None
        beneficiary_address = None
        if record:
            beneficiary_address = record.get("beneficiary_address")
            vault_object_id = record.get("vault_object_id")

        if ih and not beneficiary_id:
            beneficiary_id = resolve_beneficiary_object_for_identity(self.profile, ih)
            if beneficiary_id and not vault_object_id:
                vault_object_id = resolve_beneficiary_vault_id(self.profile, beneficiary_id)

        mysocial_creator = None
        if beneficiary_id:
            fields = fetch_username_beneficiary_fields(self.profile, beneficiary_id)
            mysocial_creator = fields.get("owner") or fields.get("claimant_address")

        source = "unresolved"
        if ih and handle:
            source = "x_handle"
        elif record:
            source = "beneficiary_db"
        elif beneficiary_id:
            source = "chain_registry"

        return ResolvedIdentity(
            identity_hash=ih,
            x_handle=handle,
            beneficiary_address=str(beneficiary_address) if beneficiary_address else None,
            vault_object_id=str(vault_object_id) if vault_object_id else None,
            beneficiary_id=str(beneficiary_id) if beneficiary_id else None,
            mysocial_creator_address=str(mysocial_creator) if mysocial_creator else None,
            confidence=creator_confidence,
            source=source,
            creator_candidate_id=creator_candidate_id,
        )
