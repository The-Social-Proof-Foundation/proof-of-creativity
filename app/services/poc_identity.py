"""X identity verification for username beneficiary claims."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import requests
import structlog

from app.chain.move_address import canonical_registry_username, parse_identity_hash
from app.network_config import NetworkProfile, load_network_profile
from app.services.identity_verification_client import IdentityVerificationClient
from app.services.poc_chain_helpers import fetch_username_beneficiary_fields

logger = structlog.get_logger()

EVIDENCE_VERSION = 1


@dataclass(frozen=True)
class VerifiedClaim:
    attested_x_handle: str
    identity_hash: bytes
    evidence_hash: bytes
    verifier: str
    beneficiary_id: str
    wallet: str
    display_name: str = ""
    bio: str = ""
    profile_picture_url: str = ""
    cover_photo_url: str = ""


def compute_evidence_hash_v1(payload: dict[str, Any]) -> bytes:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(canonical.encode("utf-8"), digest_size=32).digest()


def evidence_hash_hex(digest: bytes) -> str:
    return "0x" + digest.hex()


def _mock_verifier_allowed(profile: NetworkProfile) -> bool:
    if os.getenv("POC_IDENTITY_VERIFIER", "").strip().lower() != "mock":
        return False
    if profile.network in ("testnet", "mainnet"):
        return os.getenv("ALLOW_MOCK_IDENTITY_VERIFIER", "").strip().lower() in ("1", "true", "yes")
    return profile.network == "localnet" and profile.chain_writes_enabled


def _legacy_oauth_allowed(profile: NetworkProfile) -> bool:
    if profile.network not in ("testnet", "mainnet"):
        return True
    return os.getenv("ALLOW_LEGACY_OAUTH_TOKEN", "").strip().lower() in ("1", "true", "yes")


def resolve_identity_verifier_mode(network: str) -> str:
    explicit = os.getenv("POC_IDENTITY_VERIFIER", "").strip().lower()
    if explicit:
        return explicit
    profile = load_network_profile(network)
    if profile.network in ("testnet", "mainnet"):
        return "myso-identity"
    return "mock"


class IdentityVerifier:
    def __init__(self, network: str) -> None:
        self.network = network
        self.profile = load_network_profile(network)
        self.mode = resolve_identity_verifier_mode(network)
        self.identity_client = IdentityVerificationClient()

    def verify_claim(
        self,
        *,
        beneficiary_id: str,
        wallet: str,
        identity_hash: str | bytes,
        attested_x_handle: str | None = None,
        oauth_token: str | None = None,
        session_jwt: str | None = None,
        mock_headers: dict[str, str] | None = None,
        display_name: str = "",
        bio: str = "",
        profile_picture_url: str = "",
        cover_photo_url: str = "",
    ) -> VerifiedClaim:
        fields = fetch_username_beneficiary_fields(self.profile, beneficiary_id)
        status = fields.get("status")
        if status not in (None, "ACTIVE", 0, "0"):
            raise ValueError(f"Beneficiary {beneficiary_id} is not ACTIVE (status={status})")

        required_handle = canonical_registry_username(
            str(fields.get("required_x_handle") or fields.get("username") or "")
        )
        ih_bytes = parse_identity_hash(identity_hash)
        on_chain_hash = fields.get("creator_identity_hash")
        if on_chain_hash:
            chain_bytes = parse_identity_hash(str(on_chain_hash))
            if chain_bytes != ih_bytes:
                raise ValueError("identity_hash does not match on-chain creator_identity")

        if self.mode == "mock":
            if not _mock_verifier_allowed(self.profile):
                raise RuntimeError("Mock identity verifier is disabled for this network")
            headers = mock_headers or {}
            handle = attested_x_handle or headers.get("X-PoC-Mock-Handle") or headers.get("x-poc-mock-handle")
            if not handle:
                handle = required_handle or str(fields.get("username") or "")
            handle = canonical_registry_username(handle.strip().lstrip("@"))
            mock_hash = headers.get("X-PoC-Mock-Identity-Hash") or headers.get("x-poc-mock-identity-hash")
            if mock_hash:
                mock_bytes = parse_identity_hash(mock_hash)
                if mock_bytes != ih_bytes:
                    raise ValueError("mock identity hash mismatch")
            if required_handle and handle != required_handle:
                raise ValueError(
                    f"attested_x_handle {handle!r} does not match required_x_handle {required_handle!r}"
                )
            verifier = "mock"
            evidence_payload = {
                "v": EVIDENCE_VERSION,
                "beneficiary_id": beneficiary_id,
                "identity_source": 1,
                "identity_hash": "0x" + ih_bytes.hex(),
                "attested_x_handle": handle,
                "wallet": wallet,
                "verified_at": int(time.time()),
                "verifier": verifier,
            }
            evidence_hash = compute_evidence_hash_v1(evidence_payload)
        elif self.mode in ("myso-identity", "myso_identity", "identity-verification"):
            if oauth_token and not _legacy_oauth_allowed(self.profile):
                raise ValueError(
                    "raw oauth_token is disabled in production; use session JWT with myso-identity-verification"
                )
            if not session_jwt and not self.identity_client.service_secret:
                raise ValueError("session JWT or service secret required for myso-identity verification")
            attestation = self.identity_client.attest_for_claim_sync(
                identity_hash="0x" + ih_bytes.hex(),
                beneficiary_id=beneficiary_id,
                wallet=wallet,
                session_jwt=session_jwt,
            )
            handle = canonical_registry_username(attestation.attested_x_handle.strip().lstrip("@"))
            if required_handle and handle != required_handle:
                raise ValueError(
                    f"OAuth username {handle!r} does not match required_x_handle {required_handle!r}"
                )
            attestation_bytes = parse_identity_hash(attestation.identity_hash)
            if attestation_bytes != ih_bytes:
                raise ValueError("identity verification service returned mismatched identity_hash")
            verifier = attestation.verifier
            evidence_hash = parse_identity_hash(attestation.evidence_hash)
        elif self.mode in ("x-oauth", "x_oauth", "oauth"):
            if not _legacy_oauth_allowed(self.profile):
                raise ValueError("x-oauth mode is disabled; set POC_IDENTITY_VERIFIER=myso-identity")
            handle = self._verify_x_oauth(oauth_token=oauth_token)
            handle = canonical_registry_username(handle.strip().lstrip("@"))
            if required_handle and handle != required_handle:
                raise ValueError(
                    f"OAuth username {handle!r} does not match required_x_handle {required_handle!r}"
                )
            verifier = "x-oauth"
            evidence_payload = {
                "v": EVIDENCE_VERSION,
                "beneficiary_id": beneficiary_id,
                "identity_source": 1,
                "identity_hash": "0x" + ih_bytes.hex(),
                "attested_x_handle": handle,
                "wallet": wallet,
                "verified_at": int(time.time()),
                "verifier": verifier,
            }
            evidence_hash = compute_evidence_hash_v1(evidence_payload)
        else:
            raise RuntimeError(f"Unsupported identity verifier mode: {self.mode}")

        logger.info(
            "Identity claim verified",
            network=self.network,
            verifier=verifier,
            beneficiary_id=beneficiary_id,
            handle=handle,
        )
        return VerifiedClaim(
            attested_x_handle=handle,
            identity_hash=ih_bytes,
            evidence_hash=evidence_hash,
            verifier=verifier,
            beneficiary_id=beneficiary_id,
            wallet=wallet,
            display_name=display_name,
            bio=bio,
            profile_picture_url=profile_picture_url,
            cover_photo_url=cover_photo_url,
        )

    def _verify_x_oauth(self, *, oauth_token: str | None) -> str:
        token = (oauth_token or os.getenv("X_API_BEARER") or "").strip()
        if not token:
            raise ValueError("OAuth token required for x-oauth identity verification")
        response = requests.get(
            "https://api.twitter.com/2/users/me",
            headers={"Authorization": f"Bearer {token}"},
            params={"user.fields": "username"},
            timeout=15.0,
        )
        response.raise_for_status()
        body = response.json()
        username = (body.get("data") or {}).get("username")
        if not username:
            raise ValueError("X OAuth response missing username")
        return str(username)
