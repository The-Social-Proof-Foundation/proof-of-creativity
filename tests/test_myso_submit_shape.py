"""Guardrails: MoveCall shape for PoC submission (ABI drift detection)."""

from unittest.mock import MagicMock, patch

import pytest

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
