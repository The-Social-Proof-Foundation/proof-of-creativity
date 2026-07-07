"""Unit tests for GraphQL PoC session refresh."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from app.chain.graphql_session import (
    POC_OBJECT_ALIASES,
    build_poc_session_query,
    extract_package_address_from_move_type,
    refresh_poc_session_objects,
    resolve_platform_package_address,
)
from app.network_config import NetworkProfile, bootstrap_network_session


def test_build_poc_session_query_uses_platform_package():
    query = build_poc_session_query("0x50c1")
    assert "0x50c1::proof_of_creativity::PoCConfig" in query
    assert "0x50c1::social_proof_tokens::TokenRegistry" in query
    assert "query PoCOracleSessionObjects" in query


def test_extract_package_address_from_move_type():
    assert (
        extract_package_address_from_move_type("0xabc::proof_of_creativity::PoCConfig")
        == "0xabc"
    )
    assert extract_package_address_from_move_type("invalid") is None


@patch("app.chain.graphql_session.resolve_package_id_via_rpc")
@patch("app.chain.graphql_session._graphql_post")
def test_refresh_poc_session_objects_sets_env(mock_post, mock_package_rpc):
    mock_post.return_value = {
        "pocConfig": {"nodes": [{"address": "0xconfig"}]},
        "pocRegistry": {"nodes": [{"address": "0xregistry"}]},
        "pocVaultDirectory": {"nodes": [{"address": "0xvault"}]},
        "tokenRegistry": {"nodes": [{"address": "0xtoken"}]},
        "usernameRegistry": {"nodes": []},
        "pocUsernameBeneficiaryDirectory": {"nodes": []},
    }
    mock_package_rpc.return_value = "0xpackage"

    profile = NetworkProfile(
        network="localnet",
        rpc_url="http://127.0.0.1:9000",
        grpc_url="http://127.0.0.1:9000",
        platform_package_address="0x50c1",
    )

    env_keys = [spec[0] for spec in POC_OBJECT_ALIASES.values()] + ["MYSO_POC_PACKAGE_ID"]
    patched = {"MYSO_REFRESH_SESSION_OBJECTS": "true"}
    for key in env_keys:
        patched[key] = ""

    with patch.dict(os.environ, patched, clear=False):
        resolved = refresh_poc_session_objects(profile)
        for key in env_keys:
            if key in resolved:
                assert os.environ[key] == resolved[key]

    assert resolved["MYSO_POC_CONFIG_ID"] == "0xconfig"
    assert resolved["MYSO_POC_REGISTRY_ID"] == "0xregistry"
    assert resolved["MYSO_POC_VAULT_DIRECTORY_ID"] == "0xvault"
    assert resolved["MYSO_TOKEN_REGISTRY_ID"] == "0xtoken"
    assert resolved["MYSO_POC_PACKAGE_ID"] == "0xpackage"
    mock_package_rpc.assert_called_once_with("http://127.0.0.1:9000", "0xconfig")


@patch("app.chain.graphql_session.refresh_poc_session_objects")
def test_bootstrap_network_session_prefers_graphql_over_yaml(mock_refresh):
    mock_refresh.return_value = {
        "MYSO_POC_CONFIG_ID": "0xfromgraphql",
        "MYSO_POC_REGISTRY_ID": "0xreg",
        "MYSO_POC_VAULT_DIRECTORY_ID": "0xvault",
        "MYSO_POC_PACKAGE_ID": "0xpkg",
        "MYSO_TOKEN_REGISTRY_ID": "0xtoken",
    }

    with patch.dict(
        os.environ,
        {
            "MYSO_REFRESH_SESSION_OBJECTS": "true",
            "MYSO_POC_CONFIG_ID": "",
            "MYSO_POC_REGISTRY_ID": "",
            "MYSO_POC_VAULT_DIRECTORY_ID": "",
            "MYSO_POC_PACKAGE_ID": "",
            "MYSO_TOKEN_REGISTRY_ID": "",
        },
        clear=False,
    ):
        bootstrap_network_session("localnet")
        assert os.environ["MYSO_POC_CONFIG_ID"] == "0xfromgraphql"
        assert os.environ["MYSOCIAL_RPC_URL"] == "http://127.0.0.1:9000"


def test_resolve_platform_package_address_env_override():
    profile = NetworkProfile(
        network="localnet",
        rpc_url="http://127.0.0.1:9000",
        grpc_url="http://127.0.0.1:9000",
        platform_package_address="0x50c1",
    )
    with patch.dict(os.environ, {"MYSO_PLATFORM_PACKAGE_ADDRESS": "0xdead"}, clear=False):
        assert resolve_platform_package_address(profile) == "0xdead"
