"""Identity verifier mock gate tests."""

import os
from unittest.mock import patch

import pytest

from app.network_config import NetworkProfile
from app.services.poc_chain_helpers import normalize_beneficiary_status
from app.services.poc_identity import IdentityVerifier, compute_evidence_hash_v1


def test_normalize_beneficiary_status_active_variants():
    assert normalize_beneficiary_status(1) == "ACTIVE"
    assert normalize_beneficiary_status("1") == "ACTIVE"
    assert normalize_beneficiary_status(0) == "ACTIVE"
    assert normalize_beneficiary_status("ACTIVE") == "ACTIVE"
    assert normalize_beneficiary_status(None) == "ACTIVE"


def test_normalize_beneficiary_status_claimed_and_ended():
    assert normalize_beneficiary_status(2) == "CLAIMED"
    assert normalize_beneficiary_status("CLAIMED") == "CLAIMED"
    assert normalize_beneficiary_status(3) == "ENDED"
    assert normalize_beneficiary_status("ENDED") == "ENDED"


@patch("app.services.poc_identity.fetch_username_beneficiary_fields")
def test_mock_verifier_accepts_on_chain_status_one(mock_fields):
    mock_fields.return_value = {
        "status": 1,
        "username": "alice",
        "required_x_handle": "alice",
        "creator_identity_hash": "0xab",
        "raw": {"claimed_by": None, "claimed_at": None},
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


@patch("app.services.poc_identity.fetch_username_beneficiary_fields")
def test_mock_verifier_rejects_claimed_status(mock_fields):
    mock_fields.return_value = {
        "status": 2,
        "username": "alice",
        "required_x_handle": "alice",
        "creator_identity_hash": "0xab",
        "raw": {"claimed_by": None, "claimed_at": None},
    }
    with patch.dict(os.environ, {"POC_IDENTITY_VERIFIER": "mock"}, clear=False):
        verifier = IdentityVerifier("localnet")
        with pytest.raises(ValueError, match="not ACTIVE"):
            verifier.verify_claim(
                beneficiary_id="0xben",
                wallet="0xwallet",
                identity_hash="0xab",
                attested_x_handle="alice",
            )


@patch("app.services.poc_identity.fetch_username_beneficiary_fields")
def test_mock_verifier_rejects_already_claimed_by_field(mock_fields):
    mock_fields.return_value = {
        "status": "ACTIVE",
        "username": "alice",
        "required_x_handle": "alice",
        "creator_identity_hash": "0xab",
        "raw": {"claimed_by": "0xclaimant", "claimed_at": 123},
    }
    with patch.dict(os.environ, {"POC_IDENTITY_VERIFIER": "mock"}, clear=False):
        verifier = IdentityVerifier("localnet")
        with pytest.raises(ValueError, match="already claimed"):
            verifier.verify_claim(
                beneficiary_id="0xben",
                wallet="0xwallet",
                identity_hash="0xab",
                attested_x_handle="alice",
            )


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
