"""Unit tests for v2 gRPC checkpoint client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.chain.grpc_transport import parse_grpc_target
from app.chain.grpc_v2_client import GrpcV2Client
from app.chain.proto.myso.rpc.v2 import checkpoint_pb2, ledger_service_pb2
from app.network_config import load_network_profile


def test_parse_grpc_target_local_http():
    target, tls = parse_grpc_target("http://host.docker.internal:9000", False)
    assert target == "host.docker.internal:9000"
    assert tls is False


def test_get_chain_tip():
    profile = load_network_profile("localnet")
    client = GrpcV2Client(profile)
    stub = MagicMock()
    stub.GetServiceInfo.return_value = ledger_service_pb2.GetServiceInfoResponse(
        checkpoint_height=1234,
    )
    client._ledger = stub
    client._channel = MagicMock()
    assert client.get_chain_tip() == 1234
    assert client.last_chain_tip == 1234


def test_fetch_checkpoint():
    profile = load_network_profile("localnet")
    client = GrpcV2Client(profile)
    stub = MagicMock()
    cp = checkpoint_pb2.Checkpoint(sequence_number=7)
    stub.GetCheckpoint.return_value = ledger_service_pb2.GetCheckpointResponse(checkpoint=cp)
    client._ledger = stub
    client._channel = MagicMock()
    result = client.fetch_checkpoint(7)
    assert result is not None
    assert result.sequence_number == 7
