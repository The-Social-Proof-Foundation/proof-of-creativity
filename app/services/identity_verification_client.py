"""HTTP client for myso-identity-verification PoC claim attestation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx
import requests
import structlog

logger = structlog.get_logger()


@dataclass(frozen=True)
class PocClaimAttestation:
    attested_x_handle: str
    identity_hash: str
    evidence_hash: str
    verifier: str
    verified_at: int


@dataclass(frozen=True)
class PocClaimStatus:
    oauth_required: bool
    oauth_complete: bool
    attested_x_handle: str | None
    identity_hash: str
    wallet: str
    authorize_url: str | None = None


class IdentityVerificationClient:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or os.getenv("MYSO_IDENTITY_VERIFICATION_URL", "")).rstrip("/")
        self.service_secret = os.getenv("MYSO_IDENTITY_VERIFICATION_SERVICE_SECRET", "").strip()

    @property
    def enabled(self) -> bool:
        return bool(self.base_url)

    def _headers(self, session_jwt: str | None = None) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if session_jwt:
            headers["Authorization"] = f"Bearer {session_jwt}"
        if self.service_secret:
            headers["X-PoC-Service-Secret"] = self.service_secret
        return headers

    def attest_for_claim_sync(
        self,
        *,
        identity_hash: str,
        beneficiary_id: str,
        wallet: str,
        session_jwt: str | None = None,
    ) -> PocClaimAttestation:
        if not self.enabled:
            raise RuntimeError("MYSO_IDENTITY_VERIFICATION_URL is not configured")
        url = f"{self.base_url}/verification/poc/attest-for-claim"
        payload = {
            "identity_hash": identity_hash,
            "beneficiary_id": beneficiary_id,
            "wallet": wallet,
        }
        resp = requests.post(
            url,
            json=payload,
            headers=self._headers(session_jwt),
            timeout=15.0,
        )
        resp.raise_for_status()
        body: dict[str, Any] = resp.json()
        return PocClaimAttestation(
            attested_x_handle=str(body["attested_x_handle"]),
            identity_hash=str(body["identity_hash"]),
            evidence_hash=str(body["evidence_hash"]),
            verifier=str(body.get("verifier") or "myso-identity-verification"),
            verified_at=int(body["verified_at"]),
        )

    def get_claim_status_sync(
        self,
        *,
        identity_hash: str,
        wallet: str,
        session_jwt: str | None = None,
    ) -> PocClaimStatus:
        if not self.enabled:
            raise RuntimeError("MYSO_IDENTITY_VERIFICATION_URL is not configured")
        url = f"{self.base_url}/verification/poc/status"
        resp = requests.get(
            url,
            params={"identity_hash": identity_hash, "wallet": wallet},
            headers=self._headers(session_jwt),
            timeout=15.0,
        )
        resp.raise_for_status()
        body = resp.json()
        return PocClaimStatus(
            oauth_required=bool(body.get("oauth_required")),
            oauth_complete=bool(body.get("oauth_complete")),
            attested_x_handle=body.get("attested_x_handle"),
            identity_hash=str(body.get("identity_hash") or identity_hash),
            wallet=str(body.get("wallet") or wallet),
            authorize_url=body.get("authorize_url"),
        )

    async def get_claim_status(
        self,
        *,
        identity_hash: str,
        wallet: str,
        session_jwt: str | None = None,
    ) -> PocClaimStatus:
        if not self.enabled:
            raise RuntimeError("MYSO_IDENTITY_VERIFICATION_URL is not configured")
        url = f"{self.base_url}/verification/poc/status"
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                url,
                params={"identity_hash": identity_hash, "wallet": wallet},
                headers=self._headers(session_jwt),
            )
            resp.raise_for_status()
            body = resp.json()
        return PocClaimStatus(
            oauth_required=bool(body.get("oauth_required")),
            oauth_complete=bool(body.get("oauth_complete")),
            attested_x_handle=body.get("attested_x_handle"),
            identity_hash=str(body.get("identity_hash") or identity_hash),
            wallet=str(body.get("wallet") or wallet),
            authorize_url=body.get("authorize_url"),
        )

    async def connect_for_poc_claim(
        self,
        *,
        identity_hash: str,
        beneficiary_id: str,
        wallet: str,
        session_jwt: str,
    ) -> str:
        if not self.enabled:
            raise RuntimeError("MYSO_IDENTITY_VERIFICATION_URL is not configured")
        url = f"{self.base_url}/oauth/x/connect-for-poc-claim"
        payload = {
            "identity_hash": identity_hash,
            "beneficiary_id": beneficiary_id,
            "wallet": wallet,
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                url,
                json=payload,
                headers=self._headers(session_jwt),
            )
            resp.raise_for_status()
            body = resp.json()
        authorize_url = body.get("authorize_url")
        if not authorize_url:
            raise RuntimeError("identity verification service did not return authorize_url")
        return str(authorize_url)

    async def attest_for_claim(
        self,
        *,
        identity_hash: str,
        beneficiary_id: str,
        wallet: str,
        session_jwt: str | None = None,
    ) -> PocClaimAttestation:
        if not self.enabled:
            raise RuntimeError("MYSO_IDENTITY_VERIFICATION_URL is not configured")
        url = f"{self.base_url}/verification/poc/attest-for-claim"
        payload = {
            "identity_hash": identity_hash,
            "beneficiary_id": beneficiary_id,
            "wallet": wallet,
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                url,
                json=payload,
                headers=self._headers(session_jwt),
            )
            resp.raise_for_status()
            body: dict[str, Any] = resp.json()
        return PocClaimAttestation(
            attested_x_handle=str(body["attested_x_handle"]),
            identity_hash=str(body["identity_hash"]),
            evidence_hash=str(body["evidence_hash"]),
            verifier=str(body.get("verifier") or "myso-identity-verification"),
            verified_at=int(body["verified_at"]),
        )
