"""Round-trip tests for media asset ClaimsBundle BCS encoding."""

from app.chain.bcs_media_asset_claims import (
    compute_claims_bundle_commitment,
    decode_claims_vector_bcs,
    decode_usage_grants_vector_bcs,
    split_claims_and_grants_bcs,
)
from app.services.media_asset_submission import default_claims, default_usage_grants


def test_claims_bundle_commitment_roundtrip():
    owner = "0x" + "11" * 32
    claims = default_claims(owner)
    grants = default_usage_grants(now_ms=1_700_000_000_000)
    commitment = compute_claims_bundle_commitment(claims, grants)
    assert len(commitment) == 32

    claims_bcs, grants_bcs = split_claims_and_grants_bcs(claims, grants)
    decoded_claims = decode_claims_vector_bcs(claims_bcs)
    decoded_grants = decode_usage_grants_vector_bcs(grants_bcs)

    assert decoded_claims == claims
    assert decoded_grants == grants
    assert compute_claims_bundle_commitment(decoded_claims, decoded_grants) == commitment
