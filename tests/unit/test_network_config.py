"""Unit tests for network profile loading."""

from app.network_config import SUPPORTED_NETWORKS, load_network_profile


def test_supported_networks():
    assert "localnet" in SUPPORTED_NETWORKS
    assert "testnet" in SUPPORTED_NETWORKS
    assert "mainnet" in SUPPORTED_NETWORKS


def test_load_localnet_profile():
    profile = load_network_profile("localnet")
    assert profile.network == "localnet"
    assert profile.grpc_sync.mock_mode is False
    assert profile.grpc_sync.sync_mode == "checkpoint_v2"
    assert profile.grpc_sync.batch_size == 1000
    assert profile.grpc_sync.max_pagination_iterations == 50
    assert profile.grpc_sync.checkpoint_catchup_batch_size == 10
