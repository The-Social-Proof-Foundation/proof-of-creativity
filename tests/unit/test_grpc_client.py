"""Unit tests for GrpcChainClient sync mode selection."""

from __future__ import annotations

from app.chain.checkpoint_event_extractor import POST_CREATED_SUFFIX, is_post_created_event
from app.chain.grpc_client import (
    resolve_checkpoint_stream_id,
    resolve_event_stream_id,
    resolve_sync_mode,
    resolve_sync_start_checkpoint,
    resolve_v2_next_sequence,
    clamp_sequence_to_tip,
)
from app.network_config import GrpcSyncConfig, NetworkProfile, load_network_profile


def test_is_post_created_event_matches_post_module():
    assert is_post_created_event("post", f"0x50c1::post{POST_CREATED_SUFFIX}")


def test_is_post_created_event_rejects_other_modules():
    assert not is_post_created_event("profile", "0x50c1::profile::ProfileCreatedEvent")


def test_resolve_sync_mode_defaults_to_checkpoint_v2():
    profile = load_network_profile("localnet")
    profile.grpc_sync.mock_mode = False
    assert resolve_sync_mode(profile) == "checkpoint_v2"


def test_resolve_sync_mode_mock():
    profile = NetworkProfile(
        network="localnet",
        rpc_url="http://localhost:9000",
        grpc_url="http://localhost:9000",
        grpc_sync=GrpcSyncConfig(mock_mode=True, sync_mode="checkpoint_v2"),
    )
    assert resolve_sync_mode(profile) == "mock"


def test_resolve_stream_ids_short_db_and_padded_grpc(monkeypatch):
    monkeypatch.setenv(
        "MYSO_POC_PACKAGE_ID",
        "0x00000000000000000000000000000000000000000000000000000000000050c1",
    )
    profile = load_network_profile("localnet")
    assert resolve_checkpoint_stream_id(profile) == "0x50c1"
    assert len(resolve_event_stream_id(profile)) == 66


def test_resolve_sync_start_checkpoint_prefers_saved_over_configured():
    assert resolve_sync_start_checkpoint(
        saved_checkpoint_sequence=100,
        configured_start=856,
    ) == 100


def test_resolve_sync_start_checkpoint_uses_configured_when_no_saved():
    assert resolve_sync_start_checkpoint(
        saved_checkpoint_sequence=None,
        configured_start=856,
    ) == 856


def test_resolve_sync_start_checkpoint_zero_saved_is_valid():
    assert resolve_sync_start_checkpoint(
        saved_checkpoint_sequence=0,
        configured_start=856,
    ) == 0


def test_resolve_v2_next_sequence_from_saved_cursor():
    assert resolve_v2_next_sequence(100, None) == 101


def test_resolve_v2_next_sequence_from_configured_when_fresh():
    assert resolve_v2_next_sequence(0, 856) == 856


def test_resolve_v2_next_sequence_defaults_to_one():
    assert resolve_v2_next_sequence(0, None) == 1


def test_clamp_sequence_to_tip_resets_when_ahead():
    assert clamp_sequence_to_tip(43111, 3634) == 1


def test_clamp_sequence_to_tip_keeps_when_behind():
    assert clamp_sequence_to_tip(100, 3634) == 100
