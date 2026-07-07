"""Identity verifier mock gate tests."""

import os
from unittest.mock import patch

import pytest

from app.network_config import NetworkProfile
from app.services.poc_identity import IdentityVerifier, compute_evidence_hash_v1


def test_evidence_hash_v1_stable():
    payload = {"v": 1, "beneficiary_id": "0x1", "wallet": "0x2"}
    h1 = compute_evidence_hash_v1(payload)
    h2 = compute_evidence_hash_v1(payload)
    assert h1 == h2
    assert len(h1) == 32


@patch("app.services.poc_identity.fetch_username_beneficiary_fields")
def test_mock_verifier_rejects_wrong_handle(mock_fields):
    mock_fields.return_value = {
        "status": "ACTIVE",
        "username": "alice",
        "required_x_handle": "alice",
        "creator_identity_hash": "0xab",
    }
    profile = NetworkProfile(
        network="testnet",
        rpc_url="http://127.0.0.1:9000",
        grpc_url="http://127.0.0.1:9000",
        chain_writes_enabled=False,
    )
    with patch.dict(os.environ, {"POC_IDENTITY_VERIFIER": "mock"}, clear=False):
        verifier = IdentityVerifier("testnet")
        verifier.profile = profile
        with pytest.raises(RuntimeError, match="Mock identity verifier is disabled"):
            verifier.verify_claim(
                beneficiary_id="0xben",
                wallet="0xwallet",
                identity_hash="0xab",
                attested_x_handle="bob",
            )


@patch("app.services.poc_identity.fetch_username_beneficiary_fields")
def test_mock_verifier_localnet_ok(mock_fields):
    mock_fields.return_value = {
        "status": "ACTIVE",
        "username": "alice",
        "required_x_handle": "alice",
        "creator_identity_hash": "0xab",
    }
    with patch.dict(os.environ, {"POC_IDENTITY_VERIFIER": "mock"}, clear=False):
        verifier = IdentityVerifier("localnet")
        verified = verifier.verify_claim(
            beneficiary_id="0xben",
            wallet="0xwallet",
            identity_hash="0xab",
            attested_x_handle="alice",
        )
        assert verified.attested_x_handle == "alice"
        assert verified.verifier == "mock"
