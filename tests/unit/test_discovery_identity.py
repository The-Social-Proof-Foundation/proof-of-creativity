"""Tests for corpus identity helpers."""

from app.discovery.identity import identity_hash_from_x_handle, resolve_identity_hash


def test_identity_hash_from_x_handle():
    h = identity_hash_from_x_handle("CreatorName")
    assert h.startswith("0x")
    assert len(h) > 10


def test_resolve_identity_hash_explicit():
    assert resolve_identity_hash(None, explicit="0xdead") == "0xdead"


def test_resolve_identity_hash_from_handle():
    h = resolve_identity_hash("CreatorName")
    assert h and h.startswith("0x")
