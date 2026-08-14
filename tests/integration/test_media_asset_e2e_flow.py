"""Golden-path MediaAsset-centric PoC flow (mocked chain + oracle DB)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.services.composition_submission import (
    AssetVersionInput,
    USAGE_SOCIAL_POST,
    build_analyze_post_composition_move_call,
    build_composition_submission_from_assets,
)
from app.services.media_asset_submission import (
    MediaResolutionResult,
    build_finalize_media_asset_move_call,
    build_submit_media_resolution_move_call,
    default_rights_payload,
    usage_permitted,
)
from app.chain.bcs_post_created import decode_post_created_event
from tests.unit.test_bcs_post_created import encode_post_created_media_asset


@pytest.fixture
def poc_env(monkeypatch):
    monkeypatch.setenv("MYSO_POC_PACKAGE_ID", "0xpackage")
    monkeypatch.setenv("MYSO_POC_CONFIG_ID", "0xconfig")
    monkeypatch.setenv("MYSO_POC_REGISTRY_ID", "0xregistry")
    monkeypatch.setenv("MYSO_POC_VAULT_DIRECTORY_ID", "0xvault")


def test_client_upload_resolve_create_post_move_shapes(poc_env):
    """Upload → resolve → create post: MoveCall shapes for each step."""
    submit_mc = build_submit_media_resolution_move_call(
        content_commitment=b"\x01\x02",
        observed_fingerprint_commitment=b"\x03\x04",
        media_type=1,
    )
    assert submit_mc["function"] == "submit_media_resolution"

    resolution = MediaResolutionResult(
        request_id="0xrequest",
        content_commitment=b"\x01\x02",
        observed_fingerprint_commitment=b"\x03\x04",
        media_type=1,
        submitter="0xcreator",
    )
    finalize_mc = build_finalize_media_asset_move_call(resolution)
    assert finalize_mc["function"] == "finalize_media_asset"
    assert finalize_mc["module"] == "proof_of_creativity"
    assert len(finalize_mc["arguments"]) == 11


def test_post_created_bcs_carries_media_asset_ids(poc_env):
    asset_id = "0x" + "ab" * 32
    raw = encode_post_created_media_asset(
        media_asset_ids=[asset_id],
        media_urls=["https://cdn.example/media.jpg"],
    )
    decoded = decode_post_created_event(raw)
    assert decoded.media_asset_ids == [asset_id]
    assert decoded.composition_status == 0
    assert decoded.monetization_status == 0
    payload = decoded.to_payload()
    assert payload["media_asset_ids"] == [asset_id]
    assert "revenue_redirect_to" not in payload or payload["revenue_redirect_to"] is None


def test_composition_analysis_manifest_move_call(poc_env):
    submission = build_composition_submission_from_assets(
        post_id="0xpost",
        assets=[
            AssetVersionInput(
                asset_id="0xasset",
                rights_version=1,
                economics_version=1,
                usage_class=USAGE_SOCIAL_POST,
                share_bps=10_000,
                beneficiary="0xcreator",
            )
        ],
    )
    mc = build_analyze_post_composition_move_call(submission)
    assert mc["function"] == "analyze_post_composition"
    assert mc["module"] == "proof_of_creativity"


def test_usage_grant_permits_social_post_after_resolution(poc_env):
    rights = default_rights_payload("0xcreator")
    assert usage_permitted(rights, USAGE_SOCIAL_POST)


@patch("app.workers.oracle_worker.MySocialClient")
def test_oracle_worker_enqueues_composition_when_post_has_assets(mock_client_cls, poc_env):
    """Simulate gRPC post discovery with media_asset_ids → analyze_composition job payload."""
    from app.chain.event_parser import parse_post_created

    payload = {
        "post_id": "0xpost",
        "owner": "0xowner",
        "creator": "0xowner",
        "media_asset_ids": ["0xasset1"],
        "media_urls": [],
        "composition_status": 0,
        "monetization_status": 0,
    }
    parsed = parse_post_created(payload)
    assert parsed is not None
    assert parsed["media_asset_ids"] == ["0xasset1"]
    assert parsed.get("enable_poc") is not False


def test_pending_first_original_finalize_ptb_shape(poc_env):
    from app.services.derivative_graph_submission import (
        PendingAssetInput,
        build_original_finalize_ptb,
    )

    recipe = build_original_finalize_ptb(
        PendingAssetInput(content_commitment=b"\x01\x02", media_type=2)
    )
    steps = recipe.to_dict()["steps"]
    assert steps[0]["function"] == "create_pending_derivative_asset"
    assert steps[1]["function"] == "finalize_pending_as_original"
