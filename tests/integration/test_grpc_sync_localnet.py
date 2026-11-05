"""Integration test for live localnet v2 gRPC sync (opt-in)."""

from __future__ import annotations

import os

import pytest

from app.chain.grpc_client import resolve_event_stream_id
from app.chain.grpc_v2_client import GrpcV2Client
from app.network_config import bootstrap_network_session, load_network_profile


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("LOCALNET_GRPC_E2E") != "1",
    reason="Set LOCALNET_GRPC_E2E=1 with a running local fullnode",
)
def test_localnet_v2_grpc_checkpoint_reachable():
    """Smoke test: v2 LedgerService responds with chain tip and checkpoint."""
    bootstrap_network_session("localnet")
    profile = load_network_profile("localnet")
    stream_id = resolve_event_stream_id(profile)
    assert stream_id, "stream_id must be set via MYSO_POC_PACKAGE_ID or config"

    client = GrpcV2Client(profile)
    try:
        tip = client.get_chain_tip()
        assert tip >= 0
        if tip > 0:
            checkpoint = client.fetch_checkpoint(tip)
            assert checkpoint is not None
    finally:
        client.close()
