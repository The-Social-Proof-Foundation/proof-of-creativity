"""MoveCall builders for Phase 4 post composition enforcement."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from app.chain.move_address import bytes_to_move_hex
from app.services.myso_client import _clock_object_id, _normalize_object_id

MEDIA_COMPONENT_UNSPECIFIED = 0
MEDIA_COMPONENT_IMAGE = 1
MEDIA_COMPONENT_VIDEO = 2
MEDIA_COMPONENT_AUDIO = 3

DENIAL_SCOPE_PLAYBACK = 1
DENIAL_SCOPE_PAYOUT = 2


@dataclass
class EmbeddedAssetBinding:
    binding_id: int
    source_asset_id: str
    usage_class: int
    stem: int
    media_component: int
    evidence_commitment: bytes | None = None


def _package_id() -> str:
    return os.getenv("MYSO_POC_PACKAGE_ID", "").strip()


def _struct_type(name: str) -> str:
    return f"{_package_id()}::post::{name}"


def encode_embedded_binding(binding: EmbeddedAssetBinding) -> dict[str, Any]:
    evidence = (
        {"Some": bytes_to_move_hex(binding.evidence_commitment)}
        if binding.evidence_commitment
        else {"None": None}
    )
    return {
        "type": _struct_type("EmbeddedAssetBinding"),
        "fields": {
            "binding_id": str(int(binding.binding_id)),
            "source_asset_id": _normalize_object_id(binding.source_asset_id),
            "usage_class": str(int(binding.usage_class)),
            "stem": str(int(binding.stem)),
            "media_component": str(int(binding.media_component)),
            "evidence_commitment": evidence,
        },
    }


def _oracle_address() -> str:
    addr = os.getenv("MYSO_POC_ORACLE_ADDRESS", "").strip()
    if addr:
        return _normalize_object_id(addr)
    config_id = os.getenv("MYSO_POC_CONFIG_ID", "").strip()
    if config_id:
        return _normalize_object_id(config_id)
    return ""


def build_record_embedded_bindings_move_call(
    *,
    oracle_address: str | None = None,
    post_id: str,
    bindings: list[EmbeddedAssetBinding],
    clock_id: str | None = None,
) -> dict[str, Any]:
    return {
        "packageObjectId": _package_id(),
        "module": "post",
        "function": "record_embedded_bindings",
        "typeArguments": [],
        "arguments": [
            _normalize_object_id(oracle_address or _oracle_address()),
            _normalize_object_id(post_id),
            [encode_embedded_binding(b) for b in bindings],
            clock_id or _clock_object_id(),
        ],
        "label": "record_embedded_bindings",
    }


def build_refresh_post_asset_usage_decision_move_call(
    *,
    oracle_address: str | None = None,
    post_id: str,
    source_asset_id: str,
    binding_id: int,
    clock_id: str | None = None,
) -> dict[str, Any]:
    return {
        "packageObjectId": _package_id(),
        "module": "post",
        "function": "refresh_post_asset_usage_decision",
        "typeArguments": [],
        "arguments": [
            _normalize_object_id(oracle_address or _oracle_address()),
            _normalize_object_id(post_id),
            _normalize_object_id(source_asset_id),
            str(int(binding_id)),
            clock_id or _clock_object_id(),
        ],
        "label": "refresh_usage_decision",
    }


def build_refresh_post_usage_decisions_ptb(
    *,
    oracle_address: str | None = None,
    post_id: str,
    bindings: list[EmbeddedAssetBinding],
) -> list[dict[str, Any]]:
    return [
        build_refresh_post_asset_usage_decision_move_call(
            oracle_address=oracle_address,
            post_id=post_id,
            source_asset_id=b.source_asset_id,
            binding_id=b.binding_id,
        )
        for b in bindings
    ]


def build_submit_candidate_revenue_manifest_move_call(
    *,
    oracle_address: str | None = None,
    post_id: str,
    manifest_entries: list[dict[str, Any]],
    manifest_version: int = 1,
    clock_id: str | None = None,
) -> dict[str, Any]:
    """Build Move call for post::submit_candidate_revenue_manifest."""
    entry_type = f"{_package_id()}::media_asset::ManifestEntry"
    manifest_type = f"{_package_id()}::media_asset::RevenueManifest"
    encoded_entries = []
    for entry in manifest_entries:
        source = entry.get("source_asset_id")
        source_field = (
            {"Some": _normalize_object_id(str(source))}
            if source
            else {"None": None}
        )
        encoded_entries.append(
            {
                "type": entry_type,
                "fields": {
                    "beneficiary": _normalize_object_id(str(entry.get("beneficiary", "0x0"))),
                    "share_bps": str(int(entry.get("share_bps", 0))),
                    "source_asset_id": source_field,
                    "payout_mode": str(int(entry.get("payout_mode", 0))),
                },
            }
        )
    manifest = {
        "type": manifest_type,
        "fields": {"entries": encoded_entries},
    }
    return {
        "packageObjectId": _package_id(),
        "module": "post",
        "function": "submit_candidate_revenue_manifest",
        "typeArguments": [],
        "arguments": [
            _normalize_object_id(oracle_address or _oracle_address()),
            _normalize_object_id(post_id),
            manifest,
            str(int(manifest_version)),
            clock_id or _clock_object_id(),
        ],
        "label": "submit_candidate_revenue_manifest",
    }


@dataclass
class CompositionBindingAnalysis:
    bindings: list[EmbeddedAssetBinding] = field(default_factory=list)

    def binding_ids(self) -> list[int]:
        return [b.binding_id for b in self.bindings]
