"""Guardrails: MoveCall shape for PoC submission (ABI drift detection)."""

from unittest.mock import MagicMock, patch

import pytest

from app.chain.move_calls import (
    build_claim_username_beneficiary_call,
    build_create_username_beneficiary_call,
)
from app.services.myso_client import MySocialClient


@pytest.fixture
def poc_env_base(monkeypatch):
    monkeypatch.setenv("MYSO_POC_PACKAGE_ID", "0xpackage")
    monkeypatch.setenv("MYSO_POC_CONFIG_ID", "0xconfig")
    monkeypatch.setenv("MYSO_POC_REGISTRY_ID", "0xregistry")
    monkeypatch.setenv("MYSO_POC_VAULT_DIRECTORY_ID", "0xvault")
    monkeypatch.setenv("MYSOCIAL_RPC_URL", "https://rpc.example")


def test_submit_poc_plain_move_call_argument_count(poc_env_base):
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
            client.submit_poc_analysis(
                "0xpost",
                media_type=2,
                highest_similarity_score=88,
                original_creator=None,
                derivative_redirection_target=0,
                embedded_audio_only_derivative=False,
                apply_explicit_outcome=False,
                explicit_poc_outcome=0,
                reasoning=None,
                evidence_urls=None,
                spt_pool_id=None,
            )

    mc = captured[0]
    assert mc["module"] == "proof_of_creativity"
    assert mc["function"] == "analyze_and_update_post"
    assert len(mc["arguments"]) == 14


def test_submit_poc_sync_move_call_argument_count(poc_env_base, monkeypatch):
    monkeypatch.setenv("MYSO_TOKEN_REGISTRY_ID", "0xtokenregistry")
    captured = []

    def fake_submit(self, move_call_data):
        captured.append(move_call_data)
        return {
            "success": True,
            "tx_hash": "0xdigest2",
            "status": {"status": "success"},
            "events": [],
        }

    wallet = MagicMock()
    wallet.get_address.return_value = "0xoracle"

    with patch.object(MySocialClient, "_submit_move_call", fake_submit):
        with patch.object(MySocialClient, "fetch_post_fields", lambda self, pid: ({}, None)):
            client = MySocialClient(wallet=wallet)
            client.submit_poc_analysis(
                "0xpost",
                media_type=2,
                highest_similarity_score=90,
                original_creator="0xcreator",
                spt_pool_id="0xpool",
            )

    mc = captured[0]
    assert mc["function"] == "analyze_and_update_post_sync_token_pool"
    assert len(mc["arguments"]) == 16


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
