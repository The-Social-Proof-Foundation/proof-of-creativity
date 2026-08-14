"""Build CompositionAnalysis, RevenueManifest, and analyze_post_composition Move args."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from app.services.media_asset_submission import MANIFEST_BPS_TOTAL
from app.services.myso_client import (
    _clock_object_id,
    _move_option_address,
    _move_option_string,
    _move_option_vector_string,
    _normalize_object_id,
)
from app.services.poc_utils import DERIVATIVE_TARGET_ESCROW, DERIVATIVE_TARGET_WALLET

# Mirrors social_contracts::media_asset
USAGE_SOCIAL_POST = 1
COMPOSITION_VERIFIED = 2
COMPOSITION_INVALID = 3
MONETIZATION_NONE = 0
MONETIZATION_ENABLED = 2
MONETIZATION_RESTRICTED = 3
PAYOUT_WALLET = 0
PAYOUT_ESCROW = 1


@dataclass
class AssetVersionInput:
    asset_id: str
    rights_version: int = 1
    economics_version: int = 1
    usage_class: int = USAGE_SOCIAL_POST
    share_bps: int = 0
    beneficiary: str | None = None
    payout_mode: int = PAYOUT_WALLET
    source_asset_id: str | None = None


@dataclass
class CompositionSubmission:
    post_id: str
    composition_status: int
    monetization_status: int
    analysis: dict[str, Any]
    manifest: dict[str, Any] | None
    reasoning: str | None
    evidence_urls: list[str] | None
    contains_derivatives: bool
    contains_unresolved_assets: bool
    spt_pool_id: str | None = None


def _package_id() -> str:
    return os.getenv("MYSO_POC_PACKAGE_ID", "").strip()


def _struct_type(name: str) -> str:
    return f"{_package_id()}::media_asset::{name}"


def encode_asset_version_commitment(asset: AssetVersionInput) -> dict[str, Any]:
    return {
        "type": _struct_type("AssetVersionCommitment"),
        "fields": {
            "asset_id": _normalize_object_id(asset.asset_id),
            "rights_version": str(int(asset.rights_version)),
            "economics_version": str(int(asset.economics_version)),
            "usage_class": int(asset.usage_class),
        },
    }


def build_composition_analysis(
    assets: list[AssetVersionInput],
    *,
    usage_context: int = USAGE_SOCIAL_POST,
    analyzed_at_ms: int | None = None,
) -> dict[str, Any]:
    return {
        "type": _struct_type("CompositionAnalysis"),
        "fields": {
            "analyzed_at": str(analyzed_at_ms or int(time.time() * 1000)),
            "usage_context": int(usage_context),
            "assets": [encode_asset_version_commitment(a) for a in assets],
        },
    }


def build_revenue_manifest(
    entries: list[AssetVersionInput],
    *,
    derivative_redirection_target: int = DERIVATIVE_TARGET_WALLET,
    max_embedded_asset_redirect_bps: int | None = None,
) -> dict[str, Any]:
    """Split the creator-attributable pool across manifest entries (must sum to 10_000 bps)."""
    cap = MANIFEST_BPS_TOTAL
    if max_embedded_asset_redirect_bps is not None:
        cap = min(int(max_embedded_asset_redirect_bps), MANIFEST_BPS_TOTAL)

    manifest_entries: list[dict[str, Any]] = []
    positive = [e for e in entries if int(e.share_bps) > 0]
    if not positive:
        return {"type": _struct_type("RevenueManifest"), "fields": {"entries": []}}

    total = sum(int(e.share_bps) for e in positive)
    if total != MANIFEST_BPS_TOTAL:
        # Normalize proportionally then fix rounding on last entry.
        scaled: list[tuple[AssetVersionInput, int]] = []
        running = 0
        for i, entry in enumerate(positive):
            if i == len(positive) - 1:
                bps = MANIFEST_BPS_TOTAL - running
            else:
                bps = int(round(int(entry.share_bps) * MANIFEST_BPS_TOTAL / total))
                running += bps
            scaled.append((entry, bps))
        positive_bps = scaled
    else:
        positive_bps = [(e, int(e.share_bps)) for e in positive]

    payout_mode = PAYOUT_ESCROW if derivative_redirection_target == DERIVATIVE_TARGET_ESCROW else PAYOUT_WALLET
    reclaimed = 0
    clamped: list[tuple[AssetVersionInput, int]] = []
    for entry, bps in positive_bps:
        source_id = entry.source_asset_id or entry.asset_id
        if source_id and bps > cap:
            reclaimed += bps - cap
            bps = cap
        clamped.append((entry, bps))

    if reclaimed > 0:
        for i, (entry, bps) in enumerate(clamped):
            if not (entry.source_asset_id or entry.asset_id):
                clamped[i] = (entry, bps + reclaimed)
                reclaimed = 0
                break
        if reclaimed > 0 and clamped:
            entry, bps = clamped[0]
            clamped[0] = (entry, bps + reclaimed)

    for entry, bps in clamped:
        beneficiary = entry.beneficiary or entry.asset_id
        manifest_entries.append(
            {
                "type": _struct_type("ManifestEntry"),
                "fields": {
                    "beneficiary": _normalize_object_id(str(beneficiary)),
                    "share_bps": str(bps),
                    "source_asset_id": _move_option_address(entry.source_asset_id or entry.asset_id),
                    "payout_mode": int(entry.payout_mode or payout_mode),
                },
            }
        )

    return {"type": _struct_type("RevenueManifest"), "fields": {"entries": manifest_entries}}


def build_analyze_post_composition_move_call(
    submission: CompositionSubmission,
    *,
    config_id: str | None = None,
    registry_id: str | None = None,
    vault_directory_id: str | None = None,
    token_registry_id: str | None = None,
    clock_id: str | None = None,
) -> dict[str, Any]:
    cfg = config_id or os.getenv("MYSO_POC_CONFIG_ID", "")
    registry = registry_id or os.getenv("MYSO_POC_REGISTRY_ID", "")
    vault_dir = vault_directory_id or os.getenv("MYSO_POC_VAULT_DIRECTORY_ID", "")
    token_reg = token_registry_id or os.getenv("MYSO_TOKEN_REGISTRY_ID", "")

    tail = [
        int(submission.composition_status),
        int(submission.monetization_status),
        submission.analysis,
        [submission.manifest] if submission.manifest else [],
        _move_option_string(submission.reasoning),
        _move_option_vector_string(submission.evidence_urls),
        bool(submission.contains_derivatives),
        bool(submission.contains_unresolved_assets),
        clock_id or _clock_object_id(),
    ]

    sync_ok = submission.spt_pool_id and token_reg and registry
    if sync_ok:
        return {
            "packageObjectId": _package_id(),
            "module": "proof_of_creativity",
            "function": "analyze_post_composition_sync_token_pool",
            "typeArguments": [],
            "arguments": [
                cfg,
                registry,
                token_reg,
                vault_dir,
                _normalize_object_id(submission.post_id),
                _normalize_object_id(submission.spt_pool_id),
                *tail,
            ],
        }

    return {
        "packageObjectId": _package_id(),
        "module": "proof_of_creativity",
        "function": "analyze_post_composition",
        "typeArguments": [],
        "arguments": [
            cfg,
            registry,
            vault_dir,
            _normalize_object_id(submission.post_id),
            *tail,
        ],
    }


def build_composition_submission_from_assets(
    *,
    post_id: str,
    assets: list[AssetVersionInput],
    manifest_entries: list[AssetVersionInput] | None = None,
    derivative_redirection_target: int = DERIVATIVE_TARGET_WALLET,
    max_embedded_asset_redirect_bps: int | None = None,
    contains_derivatives: bool = False,
    contains_unresolved_assets: bool = False,
    reasoning: str | None = None,
    evidence_urls: list[str] | None = None,
    spt_pool_id: str | None = None,
) -> CompositionSubmission:
    analysis = build_composition_analysis(assets)
    manifest_src = manifest_entries or [a for a in assets if int(a.share_bps) > 0]
    manifest = build_revenue_manifest(
        manifest_src,
        derivative_redirection_target=derivative_redirection_target,
        max_embedded_asset_redirect_bps=max_embedded_asset_redirect_bps,
    )

    composition_status = COMPOSITION_VERIFIED
    monetization_status = MONETIZATION_NONE
    if contains_unresolved_assets:
        composition_status = COMPOSITION_INVALID
        monetization_status = MONETIZATION_RESTRICTED
    elif manifest_src and any(int(e.share_bps) > 0 for e in manifest_src):
        monetization_status = MONETIZATION_ENABLED

    return CompositionSubmission(
        post_id=post_id,
        composition_status=composition_status,
        monetization_status=monetization_status,
        analysis=analysis,
        manifest=manifest if manifest_src else None,
        reasoning=reasoning,
        evidence_urls=evidence_urls,
        contains_derivatives=contains_derivatives,
        contains_unresolved_assets=contains_unresolved_assets,
        spt_pool_id=spt_pool_id,
    )
