"""Guardrails: MoveCall shape for PoC submission (ABI drift detection)."""

from unittest.mock import MagicMock, patch

import pytest

from app.chain.move_calls import (
    build_claim_username_beneficiary_call,
    build_create_username_beneficiary_call,
)
from app.services.composition_submission import (
    AssetVersionInput,
    build_analyze_post_composition_move_call,
    build_composition_submission_from_assets,
)
from app.services.media_asset_submission import MediaResolutionResult, build_finalize_media_asset_move_call
from app.services.myso_client import MySocialClient


@pytest.fixture
def poc_env_base(monkeypatch):
    monkeypatch.setenv("MYSO_POC_PACKAGE_ID", "0xpackage")
    monkeypatch.setenv("MYSO_POC_CONFIG_ID", "0xconfig")
    monkeypatch.setenv("MYSO_POC_REGISTRY_ID", "0xregistry")
    monkeypatch.setenv("MYSO_POC_VAULT_DIRECTORY_ID", "0xvault")
    monkeypatch.setenv("MYSOCIAL_RPC_URL", "https://rpc.example")


def test_submit_finalize_media_asset_move_call_argument_count(poc_env_base):
    resolution = MediaResolutionResult(
        request_id="0xrequest",
        content_commitment=b"\x01\x02",
        observed_fingerprint_commitment=b"\x03\x04",
        media_type=1,
        submitter="0xsubmitter",
    )
    mc = build_finalize_media_asset_move_call(resolution)
    assert mc["module"] == "proof_of_creativity"
    assert mc["function"] == "finalize_media_asset"
    assert len(mc["arguments"]) == 11


def test_submit_analyze_post_composition_plain_move_call_argument_count(poc_env_base):
    submission = build_composition_submission_from_assets(
        post_id="0xpost",
        assets=[
            AssetVersionInput(
                asset_id="0xasset",
                rights_version=1,
                economics_version=1,
            )
        ],
    )
    mc = build_analyze_post_composition_move_call(submission)
    assert mc["module"] == "proof_of_creativity"
    assert mc["function"] == "analyze_post_composition"
    assert len(mc["arguments"]) == 13


def test_submit_analyze_post_composition_sync_move_call_argument_count(poc_env_base, monkeypatch):
    monkeypatch.setenv("MYSO_TOKEN_REGISTRY_ID", "0xtokenregistry")
    submission = build_composition_submission_from_assets(
        post_id="0xpost",
        assets=[AssetVersionInput(asset_id="0xasset")],
        spt_pool_id="0xpool",
    )
    mc = build_analyze_post_composition_move_call(submission)
    assert mc["function"] == "analyze_post_composition_sync_token_pool"
    assert len(mc["arguments"]) == 15


def test_submit_analyze_post_composition_client_wrapper(poc_env_base):
    captured = []

    def fake_submit(self, move_call_data):
        captured.append(move_call_data)
        return {
            "success": True,
            "tx_hash": "0xdigest",
            "status": {"status": "success"},
            "events": [],
        }

    wallet = MagicMock()
    wallet.get_address.return_value = "0xoracle"

    submission = build_composition_submission_from_assets(
        post_id="0xpost",
        assets=[AssetVersionInput(asset_id="0xasset")],
    )
    move_call = build_analyze_post_composition_move_call(submission)

    with patch.object(MySocialClient, "_submit_move_call", fake_submit):
        client = MySocialClient(wallet=wallet)
        client.submit_analyze_post_composition(move_call)

    assert captured[0]["function"] == "analyze_post_composition"


def test_submit_poc_analysis_logs_deprecation(poc_env_base):
    """Legacy analyze_and_update_post path remains for backward compat but is deprecated."""
    captured = []

    def fake_submit(self, move_call_data):
        captured.append(move_call_data)
        return {
            "success": True,
            "tx_hash": "0xdigest",
            "status": {"status": "success"},
            "events": [],
        }

    wallet = MagicMock()
    wallet.get_address.return_value = "0xoracle"

    with patch.object(MySocialClient, "_submit_move_call", fake_submit):
        with patch.object(MySocialClient, "fetch_post_fields", lambda self, pid: ({}, None)):
            client = MySocialClient(wallet=wallet)
            result = client.submit_poc_analysis("0xpost", media_type=2, highest_similarity_score=88)

    assert captured[0]["function"] == "analyze_and_update_post"
    assert result["move_function"] == "analyze_and_update_post"


def test_submit_media_resolution_move_call(poc_env_base):
    from app.services.media_asset_submission import build_submit_media_resolution_move_call

    mc = build_submit_media_resolution_move_call(
        content_commitment=b"\xaa\xbb",
        observed_fingerprint_commitment=b"\xcc\xdd",
        media_type=1,
    )
    assert mc["module"] == "media_asset"
    assert mc["function"] == "submit_media_resolution"
    assert len(mc["arguments"]) == 4


def test_move_call_builders_argument_counts():
    create_mc = build_create_username_beneficiary_call(
        package_id="0xpackage",
        admin_cap_id="0xadmin",
        directory_id="0xdir",
        shard_id="0xshard",
        vault_directory_id="0xvaultdir",
        username_registry_id="0xur",
        username="user",
        identity_hash="0xdead",
        clock_id="0x6",
    )
    claim_mc = build_claim_username_beneficiary_call(
        package_id="0xpackage",
        poc_config_id="0xconfig",
        profile_config_id="0xpcfg",
        directory_id="0xdir",
        shard_id="0xshard",
        username_registry_id="0xur",
        memory_registry_id="0xmr",
        ai_credit_config_id="0xaic",
        beneficiary_id="0xben",
        evidence_hash=b"",
        attested_x_handle="user",
        display_name="",
        bio="",
        profile_picture_url="",
        cover_photo_url="",
        wallet="0xwallet",
        clock_id="0x6",
    )
    assert len(create_mc["arguments"]) == 10
    assert len(claim_mc["arguments"]) == 16
