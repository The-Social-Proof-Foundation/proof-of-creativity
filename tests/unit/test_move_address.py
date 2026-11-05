"""Unit tests for Move-compatible beneficiary address derivation."""

import hashlib

from app.chain.move_address import (
    IDENTITY_SOURCE_X,
    derive_beneficiary_address,
    identity_beneficiary_address,
    parse_identity_hash,
    shard_index_for_username,
)


def test_parse_identity_hash_hex_and_utf8():
    assert parse_identity_hash("786b6579") == bytes.fromhex("786b6579")
    assert parse_identity_hash("x-user-123") == b"x-user-123"


def test_identity_beneficiary_address_matches_move_algorithm():
    identity = b"x-user-123"
    data = bytes([IDENTITY_SOURCE_X]) + identity
    expected = "0x" + hashlib.blake2b(data, digest_size=32).hexdigest()
    assert identity_beneficiary_address(identity) == expected
    assert derive_beneficiary_address("x-user-123") == expected
    assert derive_beneficiary_address("0x" + identity.hex()) == expected


def test_shard_index_deterministic():
    idx_a = shard_index_for_username("offplatform")
    idx_b = shard_index_for_username("OFFPLATFORM")
    assert idx_a == idx_b
    assert 0 <= idx_a < 256
