"""Creator identity helpers for discovered provenance."""

from __future__ import annotations

from app.chain.move_address import canonical_registry_username, parse_identity_hash


def identity_hash_from_x_handle(handle: str) -> str:
    canonical = canonical_registry_username(handle.strip().lstrip("@"))
    return "0x" + parse_identity_hash(canonical).hex()


def resolve_identity_hash(
    creator_x_handle: str | None,
    explicit: str | None = None,
) -> str | None:
    if explicit:
        return explicit
    if not creator_x_handle:
        return None
    return identity_hash_from_x_handle(creator_x_handle)
