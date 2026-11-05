"""ABI guardrails for username beneficiary Move calls."""

from app.chain.move_calls import (
    build_claim_username_beneficiary_call,
    build_create_username_beneficiary_call,
)


def test_create_username_beneficiary_argument_count():
    mc = build_create_username_beneficiary_call(
        package_id="0xpkg",
        admin_cap_id="0xadmin",
        directory_id="0xdir",
        shard_id="0xshard",
        vault_directory_id="0xvaultdir",
        username_registry_id="0xur",
        username="alice",
        identity_hash="0xabc",
        required_x_handle="alice",
        clock_id="0x6",
    )
    assert mc["module"] == "proof_of_creativity"
    assert mc["function"] == "create_username_beneficiary"
    assert len(mc["arguments"]) == 10


def test_claim_username_beneficiary_argument_count():
    mc = build_claim_username_beneficiary_call(
        package_id="0xpkg",
        poc_config_id="0xconfig",
        profile_config_id="0xpcfg",
        directory_id="0xdir",
        shard_id="0xshard",
        username_registry_id="0xur",
        memory_registry_id="0xmr",
        ai_credit_config_id="0xaic",
        beneficiary_id="0xben",
        evidence_hash=b"\x01\x02",
        attested_x_handle="alice",
        display_name="Alice",
        bio="bio",
        profile_picture_url="",
        cover_photo_url="",
        wallet="0xwallet",
        clock_id="0x6",
    )
    assert mc["function"] == "claim_username_beneficiary"
    assert len(mc["arguments"]) == 16
