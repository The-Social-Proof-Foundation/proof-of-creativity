"""SLIP-0010 Ed25519 HD derivation without coincurve/bip-utils."""

from __future__ import annotations

import hashlib
import hmac
import struct


def _ser32(index: int) -> bytes:
    if index < 0 or index > 0xFFFFFFFF:
        raise ValueError(f"Invalid derivation index: {index}")
    return struct.pack(">I", index)


def _harden(index: int) -> int:
    return index + 0x80000000


def slip10_ed25519_master(seed: bytes) -> tuple[bytes, bytes]:
    digest = hmac.new(b"ed25519 seed", seed, hashlib.sha512).digest()
    return digest[:32], digest[32:]


def slip10_ed25519_child(key: bytes, chain_code: bytes, index: int) -> tuple[bytes, bytes]:
    if index < 0x80000000:
        raise ValueError("Ed25519 SLIP-0010 only supports hardened child derivation")
    payload = b"\x00" + key + _ser32(index)
    digest = hmac.new(chain_code, payload, hashlib.sha512).digest()
    return digest[:32], digest[32:]


def slip10_ed25519_derive(seed: bytes, path: list[int]) -> bytes:
    key, chain = slip10_ed25519_master(seed)
    for index in path:
        key, chain = slip10_ed25519_child(key, chain, index)
    return key


def mysocial_ed25519_path(account: int = 0, change: int = 0, address_index: int = 0) -> list[int]:
    """MySocial path: m/44'/6976'/{account}'/{change}'/{address}'"""
    return [
        _harden(44),
        _harden(6976),
        _harden(account),
        _harden(change),
        _harden(address_index),
    ]
