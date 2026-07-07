"""PoC config cache key coverage."""

from app.services.poc_utils import OFFCHAIN_DEFAULT_POC_CONFIG

MOVE_POC_CONFIG_KEYS = {
    "oracle_address",
    "image_threshold",
    "video_threshold",
    "audio_threshold",
    "revenue_redirect_percentage",
    "dispute_cost",
    "min_vote_stake",
    "max_vote_stake",
    "voting_duration_ms",
    "max_reasoning_length",
    "max_evidence_urls",
    "max_votes_per_dispute",
    "dispute_governance_registry_id",
    "claim_treasury_fee_bps",
    "max_referral_bps",
    "video_embedded_audio_redirect_bps",
    "dispute_quorum_base_stake",
    "dispute_second_round_fee_multiplier_bps",
    "dispute_second_round_quorum_multiplier_bps",
    "username_beneficiary_join_referral_bps",
    "max_disputes_per_post",
    "min_vault_deposit_amount",
    "version",
}


def test_offchain_defaults_cover_move_fields():
    assert MOVE_POC_CONFIG_KEYS.issubset(set(OFFCHAIN_DEFAULT_POC_CONFIG.keys()))


def test_offchain_defaults_include_dispute_and_vault_fields():
    assert OFFCHAIN_DEFAULT_POC_CONFIG["min_vault_deposit_amount"] == 1
    assert OFFCHAIN_DEFAULT_POC_CONFIG["max_disputes_per_post"] == 2
    assert OFFCHAIN_DEFAULT_POC_CONFIG["dispute_second_round_fee_multiplier_bps"] == 10_000
