"""Move-compatible address derivation for PoC username beneficiaries."""

from __future__ import annotations

import hashlib
import re

IDENTITY_SOURCE_X: int = 1
NUM_SHARDS: int = 256

_HEX_RE = re.compile(r"^(0x)?([0-9a-fA-F]*)$")


def parse_identity_hash(value: str | bytes) -> bytes:
    """Normalize identity_hash to raw bytes (matches Move provisioning input)."""
    if isinstance(value, bytes):
        return value
    text = str(value).strip()
    match = _HEX_RE.match(text)
    if match and match.group(2):
        hex_part = match.group(2)
        if len(hex_part) % 2 == 1:
            hex_part = "0" + hex_part
        if hex_part:
            return bytes.fromhex(hex_part)
    return text.encode("utf-8")


def identity_beneficiary_address(
    identity_hash: str | bytes,
    *,
    identity_source: int = IDENTITY_SOURCE_X,
) -> str:
    """
    Mirror Move `poc_username_beneficiary::identity_beneficiary_address`:
    blake2b256([source] || identity_hash) -> object id bytes -> address hex.
    """
    ih_bytes = parse_identity_hash(identity_hash)
    data = bytes([identity_source & 0xFF]) + ih_bytes
    digest = hashlib.blake2b(data, digest_size=32).digest()
    return "0x" + digest.hex()


def derive_beneficiary_address(identity_hash: str | bytes) -> str:
    """Public helper used by services (network-agnostic Move address)."""
    return identity_beneficiary_address(identity_hash)


def shard_index_for_username(username: str) -> int:
    """Mirror Move `shard_index_for_username` (first byte of blake2b(username) % NUM_SHARDS)."""
    canonical = username.encode("utf-8").lower()
    digest = hashlib.blake2b(canonical, digest_size=32).digest()
    return digest[0] % NUM_SHARDS


def canonical_registry_username(username: str) -> str:
    """Mirror profile::canonical_registry_username_from_bytes (lowercase)."""
    return username.strip().lower()


def bytes_to_move_hex(data: bytes) -> str:
    if not data:
        return "0x"
    return "0x" + data.hex()


def text_to_move_hex(text: str) -> str:
    return bytes_to_move_hex(text.encode("utf-8"))
