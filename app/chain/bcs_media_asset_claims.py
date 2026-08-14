"""BCS serialization for media_asset ClaimsBundle commitment (matches Move layout)."""

from __future__ import annotations

import hashlib
from typing import Any

from app.chain.bcs_post_created import BcsDecodeError, BcsReader
from app.services.myso_client import _normalize_object_id


class BcsWriter:
    def __init__(self) -> None:
        self._buf = bytearray()

    def bytes(self) -> bytes:
        return bytes(self._buf)

    def write_u8(self, value: int) -> None:
        self._buf.append(value & 0xFF)

    def write_u64(self, value: int) -> None:
        self._buf.extend(int(value).to_bytes(8, "little"))

    def write_bool(self, value: bool) -> None:
        self.write_u8(1 if value else 0)

    def write_uleb128(self, value: int) -> None:
        n = int(value)
        while True:
            byte = n & 0x7F
            n >>= 7
            if n:
                self._buf.append(byte | 0x80)
            else:
                self._buf.append(byte)
                break

    def write_address(self, address: str) -> None:
        raw = bytes.fromhex(_normalize_object_id(address).removeprefix("0x"))
        if len(raw) != 32:
            raise ValueError(f"invalid address length: {address}")
        self._buf.extend(raw)

    def write_option_address(self, address: str | None) -> None:
        if address is None:
            self.write_u8(0)
            return
        self.write_u8(1)
        self.write_address(address)

    def write_option_bytes(self, data: bytes | None) -> None:
        if data is None:
            self.write_u8(0)
            return
        self.write_u8(1)
        self.write_bytes_vector(data)

    def write_option_u64(self, value: int | None) -> None:
        if value is None:
            self.write_u8(0)
            return
        self.write_u8(1)
        self.write_u64(value)

    def write_bytes_vector(self, data: bytes) -> None:
        self.write_uleb128(len(data))
        self._buf.extend(data)

    def write_vector(self, items: list[bytes]) -> None:
        self.write_uleb128(len(items))
        for item in items:
            self._buf.extend(item)


def serialize_claim(claim: dict[str, Any]) -> bytes:
    w = BcsWriter()
    w.write_u8(int(claim["claim_type"]))
    w.write_address(str(claim["claimant"]))
    w.write_option_address(claim.get("asset_id"))
    w.write_u64(int(claim.get("rights_mask", 0)))
    w.write_u8(int(claim.get("scope", 0)))
    w.write_u8(int(claim.get("verification_status", 2)))
    evidence = claim.get("evidence_commitment")
    if isinstance(evidence, str):
        evidence = bytes.fromhex(evidence.removeprefix("0x"))
    w.write_option_bytes(evidence if isinstance(evidence, (bytes, bytearray)) else None)
    return w.bytes()


def serialize_usage_grant(grant: dict[str, Any]) -> bytes:
    w = BcsWriter()
    w.write_u8(int(grant["usage_class"]))
    w.write_u64(int(grant.get("granted_rights", 0)))
    w.write_u8(int(grant.get("license_type", 1)))
    w.write_u8(int(grant.get("compensation_type", 1)))
    w.write_u64(int(grant.get("compensation_bps", 0)))
    w.write_bool(bool(grant.get("attribution_required", False)))
    w.write_bool(bool(grant.get("derivatives_permitted", True)))
    w.write_bool(bool(grant.get("commercial_use_permitted", True)))
    w.write_u64(int(grant.get("effective_from", 0)))
    w.write_option_u64(
        int(grant["expires_at"]) if grant.get("expires_at") is not None else None
    )
    w.write_bool(bool(grant.get("revocable", True)))
    return w.bytes()


def serialize_claims_bundle(claims: list[dict[str, Any]], usage_grants: list[dict[str, Any]]) -> bytes:
    claim_bytes = [serialize_claim(c) for c in claims]
    grant_bytes = [serialize_usage_grant(g) for g in usage_grants]
    w = BcsWriter()
    w.write_vector(claim_bytes)
    w.write_vector(grant_bytes)
    return w.bytes()


def compute_claims_bundle_commitment(
    claims: list[dict[str, Any]], usage_grants: list[dict[str, Any]]
) -> bytes:
    payload = serialize_claims_bundle(claims, usage_grants)
    return hashlib.sha3_256(payload).digest()


def split_claims_and_grants_bcs(
    claims: list[dict[str, Any]], usage_grants: list[dict[str, Any]]
) -> tuple[bytes, bytes]:
    claim_bytes = [serialize_claim(c) for c in claims]
    grant_bytes = [serialize_usage_grant(g) for g in usage_grants]
    claims_writer = BcsWriter()
    claims_writer.write_vector(claim_bytes)
    grants_writer = BcsWriter()
    grants_writer.write_vector(grant_bytes)
    return claims_writer.bytes(), grants_writer.bytes()


def _read_option_bytes(reader: BcsReader) -> bytes | None:
    tag = reader.read_u8()
    if tag == 0:
        return None
    if tag != 1:
        raise BcsDecodeError(f"invalid Option tag: {tag}")
    return reader.read_vec_u8()


def _decode_claim(reader: BcsReader) -> dict[str, Any]:
    claim_type = reader.read_u8()
    claimant = reader.read_address()
    asset_id = reader.read_option_address()
    rights_mask = reader.read_u64()
    scope = reader.read_u8()
    verification_status = reader.read_u8()
    evidence_commitment = _read_option_bytes(reader)
    return {
        "claim_type": claim_type,
        "claimant": claimant,
        "asset_id": asset_id,
        "rights_mask": rights_mask,
        "scope": scope,
        "verification_status": verification_status,
        "evidence_commitment": evidence_commitment,
    }


def _decode_usage_grant(reader: BcsReader) -> dict[str, Any]:
    usage_class = reader.read_u8()
    granted_rights = reader.read_u64()
    license_type = reader.read_u8()
    compensation_type = reader.read_u8()
    compensation_bps = reader.read_u64()
    attribution_required = reader.read_bool()
    derivatives_permitted = reader.read_bool()
    commercial_use_permitted = reader.read_bool()
    effective_from = reader.read_u64()
    expires_at = reader.read_option_u64()
    revocable = reader.read_bool()
    return {
        "usage_class": usage_class,
        "granted_rights": granted_rights,
        "license_type": license_type,
        "compensation_type": compensation_type,
        "compensation_bps": compensation_bps,
        "attribution_required": attribution_required,
        "derivatives_permitted": derivatives_permitted,
        "commercial_use_permitted": commercial_use_permitted,
        "effective_from": effective_from,
        "expires_at": expires_at,
        "revocable": revocable,
    }


def decode_claims_vector_bcs(data: bytes) -> list[dict[str, Any]]:
    reader = BcsReader(data)
    count = reader.read_uleb128()
    return [_decode_claim(reader) for _ in range(count)]


def decode_usage_grants_vector_bcs(data: bytes) -> list[dict[str, Any]]:
    reader = BcsReader(data)
    count = reader.read_uleb128()
    return [_decode_usage_grant(reader) for _ in range(count)]
