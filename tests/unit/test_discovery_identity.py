"""Tests for discovery identity hash derivation."""

from app.discovery.identity import identity_hash_from_x_handle, resolve_identity_hash


def test_identity_hash_from_x_handle():
    ih = identity_hash_from_x_handle("@Alice")
    assert ih.startswith("0x")
    assert ih == identity_hash_from_x_handle("alice")


def test_resolve_identity_hash_prefers_explicit():
    explicit = "0xdeadbeef"
    assert resolve_identity_hash("alice", explicit=explicit) == explicit


def test_resolve_identity_hash_none_without_handle():
    assert resolve_identity_hash(None) is None
