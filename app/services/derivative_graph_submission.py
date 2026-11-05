"""MoveCall builders for Phase 2–5 media_asset derivative graph flows."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from app.chain.move_address import bytes_to_move_hex
from app.chain.ptb_builder import PtbRecipe, result_ref
from app.services.media_asset_submission import ASSET_KIND_VISUAL_WORK
from app.services.myso_client import _clock_object_id, _move_option_address, _normalize_object_id

RELATIONSHIP_REMIX = 1
RELATIONSHIP_SAMPLE = 2

PROPOSAL_STATUS_PENDING = 0
PROPOSAL_STATUS_ACCEPTED = 1
PROPOSAL_STATUS_REJECTED = 2
PROPOSAL_STATUS_FINALIZED = 3


def _package_id() -> str:
    return os.getenv("MYSO_POC_PACKAGE_ID", "").strip()


@dataclass
class ParentEdgeInput:
    parent_asset_id: str
    license_instance_id: str
    template_version_id: str
    relationship_type: int = RELATIONSHIP_REMIX
    evidence_commitment: bytes | None = None


@dataclass
class PendingAssetInput:
    content_commitment: bytes
    media_type: int
    asset_kind: int = ASSET_KIND_VISUAL_WORK


def build_create_pending_derivative_asset_move_call(
    pending: PendingAssetInput,
    *,
    clock_id: str | None = None,
) -> dict[str, Any]:
    return {
        "packageObjectId": _package_id(),
        "module": "media_asset",
        "function": "create_pending_derivative_asset",
        "typeArguments": [],
        "arguments": [
            bytes_to_move_hex(pending.content_commitment),
            str(int(pending.media_type)),
            str(int(pending.asset_kind)),
            clock_id or _clock_object_id(),
        ],
        "label": "create_pending",
    }


def build_add_derivative_parent_edge_to_pending_move_call(
    *,
    pending_id: str,
    parent_asset_id: str,
    license_instance_id: str,
    template_version_id: str,
    relationship_type: int = RELATIONSHIP_REMIX,
    evidence_commitment: bytes | None = None,
    clock_id: str | None = None,
) -> dict[str, Any]:
    evidence = (
        {"Some": bytes_to_move_hex(evidence_commitment)}
        if evidence_commitment
        else {"None": None}
    )
    return {
        "packageObjectId": _package_id(),
        "module": "media_asset",
        "function": "add_derivative_parent_edge_to_pending",
        "typeArguments": [],
        "arguments": [
            _normalize_object_id(pending_id),
            _normalize_object_id(parent_asset_id),
            _normalize_object_id(license_instance_id),
            _normalize_object_id(template_version_id),
            str(int(relationship_type)),
            evidence,
            clock_id or _clock_object_id(),
        ],
        "label": "add_parent_edge",
    }


def build_finalize_derivative_asset_move_call(
    pending_id: str,
    *,
    clock_id: str | None = None,
) -> dict[str, Any]:
    return {
        "packageObjectId": _package_id(),
        "module": "media_asset",
        "function": "finalize_derivative_asset",
        "typeArguments": [],
        "arguments": [
            _normalize_object_id(pending_id),
            clock_id or _clock_object_id(),
        ],
        "label": "finalize_derivative",
    }


def build_finalize_pending_as_original_move_call(
    pending_id: str,
    *,
    clock_id: str | None = None,
) -> dict[str, Any]:
    return {
        "packageObjectId": _package_id(),
        "module": "media_asset",
        "function": "finalize_pending_as_original",
        "typeArguments": [],
        "arguments": [
            _normalize_object_id(pending_id),
            clock_id or _clock_object_id(),
        ],
        "label": "finalize_original",
    }


def build_derivative_finalize_ptb(
    pending: PendingAssetInput,
    parents: list[ParentEdgeInput],
) -> PtbRecipe:
    calls: list[dict[str, Any]] = [build_create_pending_derivative_asset_move_call(pending)]
    for edge in parents:
        calls.append(
            build_add_derivative_parent_edge_to_pending_move_call(
                pending_id=result_ref(0),
                parent_asset_id=edge.parent_asset_id,
                license_instance_id=edge.license_instance_id,
                template_version_id=edge.template_version_id,
                relationship_type=edge.relationship_type,
                evidence_commitment=edge.evidence_commitment,
            )
        )
    calls.append(build_finalize_derivative_asset_move_call(result_ref(0)))
    return PtbRecipe.from_move_calls(calls, description="derivative_finalize")


def build_original_finalize_ptb(pending: PendingAssetInput) -> PtbRecipe:
    calls = [
        build_create_pending_derivative_asset_move_call(pending),
        build_finalize_pending_as_original_move_call(result_ref(0)),
    ]
    return PtbRecipe.from_move_calls(calls, description="original_finalize")


def build_materialize_initial_resolved_policy_move_call(
    asset_id: str,
    *,
    clock_id: str | None = None,
) -> dict[str, Any]:
    return {
        "packageObjectId": _package_id(),
        "module": "media_asset",
        "function": "materialize_initial_resolved_policy",
        "typeArguments": [],
        "arguments": [
            _normalize_object_id(asset_id),
            clock_id or _clock_object_id(),
        ],
        "label": "materialize_policy",
    }


def build_begin_policy_refresh_move_call(
    asset_id: str,
    *,
    clock_id: str | None = None,
) -> dict[str, Any]:
    return {
        "packageObjectId": _package_id(),
        "module": "media_asset",
        "function": "begin_policy_refresh",
        "typeArguments": [],
        "arguments": [
            _normalize_object_id(asset_id),
            clock_id or _clock_object_id(),
        ],
        "label": "begin_policy_refresh",
    }


def build_merge_parent_into_policy_refresh_move_call(
    *,
    refresh_id: str,
    parent_asset_id: str,
    template_version_id: str,
    relationship_id: int,
    clock_id: str | None = None,
) -> dict[str, Any]:
    return {
        "packageObjectId": _package_id(),
        "module": "media_asset",
        "function": "merge_parent_into_policy_refresh",
        "typeArguments": [],
        "arguments": [
            _normalize_object_id(refresh_id),
            _normalize_object_id(parent_asset_id),
            _normalize_object_id(template_version_id),
            str(int(relationship_id)),
            clock_id or _clock_object_id(),
        ],
        "label": "merge_policy_parent",
    }


def build_finalize_policy_refresh_move_call(
    asset_id: str,
    refresh_id: str,
    *,
    clock_id: str | None = None,
) -> dict[str, Any]:
    return {
        "packageObjectId": _package_id(),
        "module": "media_asset",
        "function": "finalize_policy_refresh",
        "typeArguments": [],
        "arguments": [
            _normalize_object_id(asset_id),
            _normalize_object_id(refresh_id),
            clock_id or _clock_object_id(),
        ],
        "label": "finalize_policy_refresh",
    }


def build_propose_detected_relationship_move_call(
    *,
    config_id: str,
    accused_pending_id: str,
    original_asset_id: str,
    similarity_bps: int,
    evidence_commitment: bytes | None = None,
    clock_id: str | None = None,
) -> dict[str, Any]:
    evidence = (
        {"Some": bytes_to_move_hex(evidence_commitment)}
        if evidence_commitment
        else {"None": None}
    )
    return {
        "packageObjectId": _package_id(),
        "module": "proof_of_creativity",
        "function": "propose_detected_relationship",
        "typeArguments": [],
        "arguments": [
            _normalize_object_id(config_id),
            _normalize_object_id(accused_pending_id),
            _normalize_object_id(original_asset_id),
            str(int(max(0, min(10_000, similarity_bps)))),
            evidence,
            clock_id or _clock_object_id(),
        ],
        "label": "propose_detected_relationship",
    }


def build_accept_detected_relationship_move_call(
    *,
    config_id: str,
    proposal_id: str,
    clock_id: str | None = None,
) -> dict[str, Any]:
    return {
        "packageObjectId": _package_id(),
        "module": "proof_of_creativity",
        "function": "accept_detected_relationship",
        "typeArguments": [],
        "arguments": [
            _normalize_object_id(config_id),
            _normalize_object_id(proposal_id),
            clock_id or _clock_object_id(),
        ],
        "label": "accept_detected_relationship",
    }


def build_reject_detected_relationship_move_call(
    *,
    config_id: str,
    proposal_id: str,
    clock_id: str | None = None,
) -> dict[str, Any]:
    return {
        "packageObjectId": _package_id(),
        "module": "proof_of_creativity",
        "function": "reject_detected_relationship",
        "typeArguments": [],
        "arguments": [
            _normalize_object_id(config_id),
            _normalize_object_id(proposal_id),
            clock_id or _clock_object_id(),
        ],
        "label": "reject_detected_relationship",
    }


def build_finalize_detected_lineage_move_call(
    *,
    config_id: str,
    proposal_id: str,
    pending_id: str,
    license_instance_id: str,
    template_version_id: str,
    clock_id: str | None = None,
) -> dict[str, Any]:
    return {
        "packageObjectId": _package_id(),
        "module": "proof_of_creativity",
        "function": "finalize_detected_lineage",
        "typeArguments": [],
        "arguments": [
            _normalize_object_id(config_id),
            _normalize_object_id(proposal_id),
            _normalize_object_id(pending_id),
            _normalize_object_id(license_instance_id),
            _normalize_object_id(template_version_id),
            clock_id or _clock_object_id(),
        ],
        "label": "finalize_detected_lineage",
    }


def build_accept_and_finalize_detected_ptb(
    *,
    config_id: str,
    proposal_id: str,
    pending_id: str,
    parent_asset_id: str,
    license_instance_id: str,
    template_version_id: str,
) -> PtbRecipe:
    calls = [
        build_accept_detected_relationship_move_call(
            config_id=config_id,
            proposal_id=proposal_id,
        ),
        build_add_derivative_parent_edge_to_pending_move_call(
            pending_id=pending_id,
            parent_asset_id=parent_asset_id,
            license_instance_id=license_instance_id,
            template_version_id=template_version_id,
        ),
        build_finalize_derivative_asset_move_call(pending_id),
    ]
    return PtbRecipe.from_move_calls(calls, description="accept_and_finalize_detected")
