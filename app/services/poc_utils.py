"""Shared helpers for PoC oracle ↔ Move contract (scaling, truncation, constants)."""

from __future__ import annotations

import os
from typing import Any, List, Optional

# Align with social_contracts::proof_of_creativity
MEDIA_TYPE_IMAGE: int = 1
MEDIA_TYPE_VIDEO: int = 2
MEDIA_TYPE_AUDIO: int = 3

DERIVATIVE_TARGET_WALLET: int = 0
DERIVATIVE_TARGET_ESCROW: int = 1

OUTCOME_ROYALTY_FREE: int = 4

# Mirrors on-chain defaults when RPC is unavailable (attestations / dry runs only)
OFFCHAIN_DEFAULT_POC_CONFIG = {
    "oracle_address": None,
    "image_threshold": 95,
    "video_threshold": 95,
    "audio_threshold": 95,
    "revenue_redirect_percentage": 100,
    "claim_treasury_fee_bps": 100,
    "max_referral_bps": 500,
    "video_embedded_audio_redirect_bps": 3000,
    "max_reasoning_length": 5000,
    "max_evidence_urls": 10,
    "dispute_cost": 0,
    "min_vote_stake": 1_000_000_000,
    "max_vote_stake": 100_000_000_000,
    "voting_duration_ms": 604_800_000,
    "max_votes_per_dispute": 10_000,
    "dispute_governance_registry_id": None,
    "dispute_quorum_base_stake": 0,
    "dispute_second_round_fee_multiplier_bps": 10_000,
    "dispute_second_round_quorum_multiplier_bps": 10_000,
    "username_beneficiary_join_referral_bps": 500,
    "max_disputes_per_post": 2,
    "min_vault_deposit_amount": 1,
    "max_embedded_asset_redirect_bps": 5000,
    "version": 0,
}


def redirect_target_from_env(value: Optional[str]) -> int:
    v = (value or "wallet").strip().lower()
    if v in ("escrow", "vault", "1"):
        return DERIVATIVE_TARGET_ESCROW
    return DERIVATIVE_TARGET_WALLET


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def mys_integration_enabled_from_env() -> bool:
    """True when MYSO_INTEGRATION_ENABLED is truthy (matches init_myso_client gate)."""
    return _env_flag("MYSO_INTEGRATION_ENABLED", False)


def poc_strict_oracle_from_env() -> bool:
    """If true, init_myso_client returns None when oracle wallet != PoCConfig.oracle_address."""
    return _env_flag("MYSO_POC_STRICT_ORACLE", False)


def poc_fail_lifespan_on_mys_misconfig_from_env() -> bool:
    """If true and integration is enabled, FastAPI lifespan aborts when MySocial client init fails."""
    return _env_flag("MYSO_POC_FAIL_LIFESPAN_ON_MYS_MISCONFIG", False)


def poc_require_tx_when_post_id_from_env() -> bool:
    """
    If true, uploads that include post_id while MySocial client is active must obtain a tx digest;
    RPC failures (or missing digest) fail the request instead of completing with tx_hash=null.
    """
    return _env_flag("MYSO_POC_REQUIRE_TX_WHEN_POST_ID", False)


def mysocial_readiness_payload(myso_client: Optional[Any]) -> dict:
    """
    Serializable status for /health and /readyz (myso_client is global MySocialClient or None).
    """
    requested = mys_integration_enabled_from_env()
    active = myso_client is not None
    oracle_authorized: Optional[bool] = None
    if active:
        try:
            oracle_authorized = bool(myso_client.verify_oracle_authorization())
        except Exception:
            oracle_authorized = False
    need_mys = requested
    ready = not need_mys or (active and oracle_authorized is True)
    return {
        "integration_requested": requested,
        "client_active": active,
        "oracle_authorized": oracle_authorized,
        "ready_for_submission": ready,
    }


def similarity_float_to_u64_percent(score: float) -> int:
    """
    Move compares u64 percentages 0–100 to thresholds. Map internal [0,1] float similarity
    with clamp + bankers rounding to nearest integer percent.
    """
    if score != score or score <= 0.0:
        return 0
    scaled = score * 100.0
    if scaled >= 99.995:
        return 100
    if scaled <= 0.0:
        return 0
    return int(round(score * 100.0))


def truncate_reasoning(reasoning: Optional[str], max_length: int) -> Optional[str]:
    if reasoning is None:
        return None
    txt = reasoning.strip()
    if not txt:
        return None
    if len(txt) <= max_length:
        return txt
    return txt[: max(0, max_length - 1)].rstrip() + "…"


def truncate_evidence_urls(urls: Optional[List[str]], max_count: int) -> Optional[List[str]]:
    if not urls:
        return None
    trimmed = [u.strip() for u in urls if u and str(u).strip()]
    if not trimmed:
        return None
    return trimmed[:max_count]
