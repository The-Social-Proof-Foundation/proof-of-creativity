"""
MySocial RPC Client for Proof of Creativity Oracle
Submits analysis results to MySocial blockchain
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
import structlog

from app.services.myso_wallet import MySocialWallet, load_oracle_wallet
from app.services.poc_utils import redirect_target_from_env

logger = structlog.get_logger()

POC_OUTCOME_NONE = 0


def _move_option_address(addr: Optional[str]):
    """Serialize Option<address> as used by legacy MySocial tx JSON (same pattern as earlier client)."""
    return [addr] if addr else []


def _move_option_string(val: Optional[str]):
    """Option<String>: empty vector for None."""
    return [val] if val else []


def _move_option_vector_string(urls: Optional[List[str]]):
    """Option<vector<String>>: wrap inner list for Some."""
    if not urls:
        return []
    return [urls]


def _normalize_object_id(obj_id: str) -> str:
    """Pad short object ids (e.g. 0x6) to 32-byte hex for RPC move calls."""
    raw = obj_id.strip().lower()
    if not raw.startswith("0x"):
        raw = f"0x{raw}"
    hex_part = raw[2:]
    if len(hex_part) < 64:
        hex_part = hex_part.zfill(64)
    return f"0x{hex_part}"


def _clock_object_id() -> str:
    return _normalize_object_id(os.getenv("MYSO_CLOCK_OBJECT_ID", "0x6"))


def _similarity_score_u64(score: int) -> str:
    return str(int(max(0, min(100, score))))


def _parse_po_u8(val: Any, default: int = 0) -> int:
    if val is None:
        return default
    if isinstance(val, int):
        return val
    if isinstance(val, str) and val.isdigit():
        return int(val)
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _truthy_nested_option(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, dict):
        if val.get("None") or val.get("variant") == "None":
            return False
        if "Some" in val or val.get("fields"):
            inner = val.get("Some") or val.get("fields")
            return _truthy_nested_option(inner)
        return True
    if isinstance(val, str):
        return bool(val.strip())
    if isinstance(val, (list, tuple)):
        return len(val) > 0
    return True


class MySocialClient:
    """Client for submitting PoC analysis to MySocial blockchain"""

    _config_cache_lock = threading.Lock()
    _config_cache_blob: Tuple[float, Dict[str, Any]] = (0.0, {})

    def __init__(self, wallet: Optional[MySocialWallet] = None):
        self.rpc_url = os.getenv("MYSOCIAL_RPC_URL", "https://fullnode.testnet.mysocial.io")
        self.package_id = os.getenv("MYSO_POC_PACKAGE_ID")
        self.config_id = os.getenv("MYSO_POC_CONFIG_ID")
        self.registry_id = os.getenv("MYSO_POC_REGISTRY_ID")
        self.vault_directory_id = os.getenv("MYSO_POC_VAULT_DIRECTORY_ID")
        self.token_registry_id = os.getenv("MYSO_TOKEN_REGISTRY_ID")
        self.redirect_target = redirect_target_from_env(os.getenv("MYSO_POC_REDIRECT_TARGET"))

        ttl_raw = os.getenv("MYSO_POC_CONFIG_CACHE_TTL_SECONDS", "60")
        try:
            self._ttl_seconds = float(ttl_raw)
        except ValueError:
            self._ttl_seconds = 60.0

        self.wallet = wallet or load_oracle_wallet()
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self._oracle_verify_rpc_unreachable = False

        logger.info(
            "MySocial client initialized",
            rpc_url=self.rpc_url,
            oracle_address=self.wallet.get_address(),
            package_id=self.package_id,
        )

    # --- PoC config (cached RPC) ---
    def get_poc_config(self, bypass_cache: bool = False) -> dict:
        now = time.monotonic()
        with MySocialClient._config_cache_lock:
            if (
                not bypass_cache
                and MySocialClient._config_cache_blob[1]
                and now - MySocialClient._config_cache_blob[0] < self._ttl_seconds
            ):
                cached_at, blob = MySocialClient._config_cache_blob
                return dict(blob)

        raw_fields = self._fetch_poc_config_fields()
        parsed = self._normalize_poc_config(raw_fields)

        with MySocialClient._config_cache_lock:
            MySocialClient._config_cache_blob = (now, parsed)
            return dict(parsed)

    def _normalize_poc_config(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        gov_id = fields.get("dispute_governance_registry_id")
        if isinstance(gov_id, dict):
            gov_id = gov_id.get("id") or gov_id.get("bytes")
        return {
            "oracle_address": fields.get("oracle_address"),
            "image_threshold": _parse_po_u8(fields.get("image_threshold"), 95),
            "video_threshold": _parse_po_u8(fields.get("video_threshold"), 95),
            "audio_threshold": _parse_po_u8(fields.get("audio_threshold"), 95),
            "revenue_redirect_percentage": _parse_po_u8(fields.get("revenue_redirect_percentage"), 100),
            "claim_treasury_fee_bps": _parse_po_u8(fields.get("claim_treasury_fee_bps"), 100),
            "max_referral_bps": _parse_po_u8(fields.get("max_referral_bps"), 500),
            "video_embedded_audio_redirect_bps": _parse_po_u8(fields.get("video_embedded_audio_redirect_bps"), 3000),
            "max_reasoning_length": _parse_po_u8(fields.get("max_reasoning_length"), 5000),
            "max_evidence_urls": _parse_po_u8(fields.get("max_evidence_urls"), 10),
            "dispute_cost": _parse_po_u8(fields.get("dispute_cost"), 0),
            "min_vote_stake": _parse_po_u8(fields.get("min_vote_stake"), 1_000_000_000),
            "max_vote_stake": _parse_po_u8(fields.get("max_vote_stake"), 100_000_000_000),
            "voting_duration_ms": _parse_po_u8(fields.get("voting_duration_ms"), 604_800_000),
            "max_votes_per_dispute": _parse_po_u8(fields.get("max_votes_per_dispute"), 10_000),
            "dispute_governance_registry_id": str(gov_id) if gov_id else None,
            "dispute_quorum_base_stake": _parse_po_u8(fields.get("dispute_quorum_base_stake"), 0),
            "dispute_second_round_fee_multiplier_bps": _parse_po_u8(
                fields.get("dispute_second_round_fee_multiplier_bps"), 10000
            ),
            "dispute_second_round_quorum_multiplier_bps": _parse_po_u8(
                fields.get("dispute_second_round_quorum_multiplier_bps"), 10000
            ),
            "username_beneficiary_join_referral_bps": _parse_po_u8(
                fields.get("username_beneficiary_join_referral_bps"), 500
            ),
            "max_disputes_per_post": _parse_po_u8(fields.get("max_disputes_per_post"), 2),
            "min_vault_deposit_amount": _parse_po_u8(fields.get("min_vault_deposit_amount"), 1),
            "version": _parse_po_u8(fields.get("version"), 0),
        }

    def _fetch_poc_config_fields(self) -> Dict[str, Any]:
        rpc_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "myso_getObject",
            "params": [self.config_id, {"showContent": True}],
        }
        response = self.session.post(self.rpc_url, json=rpc_request, timeout=10)
        response.raise_for_status()
        result = response.json()
        if "error" in result:
            raise RuntimeError(str(result["error"]))
        return result.get("result", {}).get("data", {}).get("content", {}).get("fields", {}) or {}

    def fetch_post_fields(self, post_id: str) -> Tuple[Dict[str, Any], Optional[Any]]:
        """
        Fetch on-chain Post object fields plus raw RPC result (may include errors).
        """
        rpc_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "myso_getObject",
            "params": [post_id, {"showContent": True}],
        }
        response = self.session.post(self.rpc_url, json=rpc_request, timeout=10)
        response.raise_for_status()
        result = response.json()
        if "error" in result:
            return {}, result["error"]
        fields = (
            result.get("result", {})
            .get("data", {})
            .get("content", {})
            .get("fields", {})
            or {}
        )
        return fields, None

    @staticmethod
    def post_has_active_poc(fields: Dict[str, Any]) -> bool:
        """
        Mirrors proof_of_creativity::has_poc_data semantics (RPC field names).

        Blocks new oracle submissions while badge / redirect snapshot exists.
        """
        if _parse_po_u8(fields.get("poc_outcome"), 0) != POC_OUTCOME_NONE:
            return True
        rr = fields.get("revenue_redirect_to")
        if rr is None:
            pass
        elif isinstance(rr, dict) and ("Some" in rr or rr.get("fields")):
            if _truthy_nested_option(rr):
                return True
        elif isinstance(rr, str) and rr.strip():
            return True
        badge_snap = fields.get("poc_badge_snapshot") or fields.get("poc_badge_snapshot ")
        badge_obj = fields.get("poc_badge_object_id")
        return _truthy_nested_option(badge_snap) or _truthy_nested_option(badge_obj)

    def check_post_already_analyzed(
        self,
        post_id: str,
        *,
        force_reanalyze: bool = False,
    ) -> dict:
        """
        Whether this post currently holds active PoC state (badge / redirect / non-none outcome).

        `force_reanalyze` is accepted for API compatibility; overturn-cleared posts have no active
        PoC and are not blocked regardless. `poc_disputes_submitted` survives clears and is echoed
        for observability only.
        """
        try:
            post_data, rpc_err = self.fetch_post_fields(post_id)
            if rpc_err:
                logger.warning("Post RPC error", post_id=post_id, error=rpc_err)
                return {
                    "already_analyzed": False,
                    "post_has_active_poc": False,
                    "poc_outcome": None,
                    "revenue_redirect_to": None,
                    "poc_badge_snapshot": None,
                    "poc_badge_object_id": None,
                    "poc_disputes_submitted": None,
                    "error": rpc_err,
                }

            poc_outcome = post_data.get("poc_outcome")
            poc_disputes = post_data.get("poc_disputes_submitted")
            rr = post_data.get("revenue_redirect_to")
            badge_snap = post_data.get("poc_badge_snapshot")
            badge_oid = post_data.get("poc_badge_object_id")
            active = self.post_has_active_poc(post_data)
            allow_force = os.getenv("MYSO_POC_ALLOW_FORCE_RESUBMIT", "").lower() in ("1", "true", "yes")
            blocking = False if (force_reanalyze and allow_force) else active

            logger.info(
                "Post PoC check",
                post_id=post_id,
                blocking=blocking,
                post_has_active_poc=active,
                force_reanalyze=force_reanalyze,
                poc_disputes_submitted=poc_disputes,
            )

            return {
                "already_analyzed": blocking,
                "post_has_active_poc": active,
                "poc_outcome": poc_outcome,
                "revenue_redirect_to": rr,
                "poc_badge_snapshot": badge_snap,
                "poc_badge_object_id": badge_oid,
                "poc_disputes_submitted": poc_disputes,
                "supports_resubmit_when_cleared": not active,
            }

        except Exception as e:
            logger.error("Failed to check post PoC status", post_id=post_id, error=str(e))
            return {
                "already_analyzed": False,
                "post_has_active_poc": False,
                "poc_outcome": None,
                "revenue_redirect_to": None,
                "poc_badge_snapshot": None,
                "poc_badge_object_id": None,
                "poc_disputes_submitted": None,
                "error": str(e),
            }

    def extract_spt_pool_id(self, fields: Dict[str, Any]) -> Optional[str]:
        sid = fields.get("spt_id")
        if isinstance(sid, dict):
            inner = sid.get("Some") or sid.get("fields")
            if isinstance(inner, dict) and inner.get("variant") == "None":
                return None
            if isinstance(inner, str):
                return inner
            if isinstance(inner, dict):
                return inner.get("id") or inner.get("bytes")
        if isinstance(sid, str) and sid.strip():
            return sid
        return None

    # --- Transactions ---
    # Move entry `analyze_and_update_post_sync_token_pool` (verify against deployed package):
    #   config, registry, token_registry, vault_directory, post, token_pool,
    #   media_type, highest_similarity_score, original_creator (Option),
    #   derivative_redirection_target, embedded_audio_only_derivative,
    #   apply_explicit_outcome, explicit_poc_outcome, reasoning (Option), evidence_urls (Option).
    # Plain `analyze_and_update_post` omits token_registry + token_pool; post is first of the tail args.
    def submit_poc_analysis(
        self,
        post_id: str,
        *,
        media_type: int,
        highest_similarity_score: int,
        original_creator: Optional[str],
        derivative_redirection_target: Optional[int] = None,
        embedded_audio_only_derivative: bool = False,
        apply_explicit_outcome: bool = False,
        explicit_poc_outcome: int = 0,
        reasoning: Optional[str] = None,
        evidence_urls: Optional[List[str]] = None,
        spt_pool_id: Optional[str] = None,
    ) -> dict:
        logger.info(
            "Submitting PoC analysis to MySocial",
            post_id=post_id,
            media_type=media_type,
            similarity_score=highest_similarity_score,
            is_derivative=original_creator is not None,
            embedded_audio_only_derivative=embedded_audio_only_derivative,
            uses_sync_variant=bool(spt_pool_id and self.token_registry_id),
        )

        fields, err = self.fetch_post_fields(post_id)
        resolved_pool = spt_pool_id or (self.extract_spt_pool_id(fields) if fields else None)
        derivative_target = derivative_redirection_target if derivative_redirection_target is not None else self.redirect_target

        tail = [
            media_type,
            _similarity_score_u64(highest_similarity_score),
            _move_option_address(original_creator),
            derivative_target,
            bool(embedded_audio_only_derivative),
            bool(apply_explicit_outcome),
            int(explicit_poc_outcome),
            _move_option_string(reasoning),
            _move_option_vector_string(evidence_urls or None),
            _clock_object_id(),
        ]

        sync_ok = resolved_pool is not None and self.token_registry_id and self.registry_id

        if sync_ok:
            data = {
                "packageObjectId": self.package_id,
                "module": "proof_of_creativity",
                "function": "analyze_and_update_post_sync_token_pool",
                "typeArguments": [],
                "arguments": [
                    self.config_id,
                    self.registry_id,
                    self.token_registry_id,
                    self.vault_directory_id,
                    post_id,
                    resolved_pool,
                    *tail,
                ],
            }
        else:
            if resolved_pool:
                logger.warning(
                    "SPT pool resolved but TokenRegistry unset; submitting without pool sync.",
                    resolved_pool_id=resolved_pool,
                )

            data = {
                "packageObjectId": self.package_id,
                "module": "proof_of_creativity",
                "function": "analyze_and_update_post",
                "typeArguments": [],
                "arguments": [
                    self.config_id,
                    self.registry_id,
                    self.vault_directory_id,
                    post_id,
                    *tail,
                ],
            }

        result = self._submit_move_call(data)
        result["move_function"] = data["function"]
        result["resolved_spt_pool_id"] = resolved_pool if sync_ok else None
        result["derivative_redirection_target"] = derivative_target
        return result

    def _build_unsigned_move_call(self, move_call_data: dict, sender: str) -> dict:
        rpc_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "unsafe_moveCall",
            "params": [
                sender,
                move_call_data["packageObjectId"],
                move_call_data["module"],
                move_call_data["function"],
                move_call_data.get("typeArguments", []),
                move_call_data["arguments"],
                None,
                os.getenv("MYSO_POC_GAS_BUDGET", "30000000"),
            ],
        }
        response = self.session.post(self.rpc_url, json=rpc_request, timeout=60)
        response.raise_for_status()
        result = response.json()
        if "error" in result:
            error_msg = result.get("error", {})
            logger.error("Failed to build unsigned transaction", error=json.dumps(error_msg, indent=2))
            raise RuntimeError(str(error_msg))
        return result["result"]

    def _submit_move_call(self, move_call_data: dict, *, wallet: Optional[MySocialWallet] = None) -> dict:
        signer = wallet or self.wallet
        built = self._build_unsigned_move_call(move_call_data, signer.get_address())
        signature_b64 = signer.sign_transaction_block_b64(built["txBytes"])
        return self._execute_transaction_bytes(built["txBytes"], signature_b64, move_call_data)

    def _execute_transaction_bytes(
        self,
        tx_bytes_b64: str,
        signature_b64: str,
        move_call_data: dict,
    ) -> dict:
        rpc_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "myso_executeTransactionBlock",
            "params": [
                tx_bytes_b64,
                [signature_b64],
                {"showInput": True, "showEffects": True, "showEvents": True},
            ],
        }
        logger.info(
            "🚀 SUBMITTING PROOF_OF_CREATIVITY TRANSACTION TO BLOCKCHAIN",
            package_id=self.package_id,
            module=move_call_data.get("module"),
            function=move_call_data.get("function"),
            rpc_endpoint=self.rpc_url,
            oracle_address=self.wallet.get_address(),
        )

        response = self.session.post(self.rpc_url, json=rpc_request, timeout=60)
        response.raise_for_status()
        result = response.json()

        if "error" in result:
            error_msg = result.get("error", {})
            logger.error("❌ PROOF_OF_CREATIVITY TRANSACTION REJECTED BY RPC", error=json.dumps(error_msg, indent=2))
            raise RuntimeError(str(error_msg))

        tx_result = result.get("result", {})
        tx_hash = tx_result.get("digest")
        tx_status = tx_result.get("effects", {}).get("status")

        logger.info(
            "✅ PROOF_OF_CREATIVITY TRANSACTION SUBMITTED SUCCESSFULLY",
            tx_hash=tx_hash,
            status=tx_status,
        )

        return {
            "success": True,
            "tx_hash": tx_hash,
            "status": tx_status,
            "events": tx_result.get("events", []),
        }

    def _submit_transaction(self, tx_data: dict, signature: str | None = None) -> dict:
        """Submit a move call using the transaction builder + BCS execute path."""
        move_call_data = tx_data.get("data", tx_data)
        return self._submit_move_call(move_call_data)

    def verify_oracle_authorization(self) -> bool:
        """Check if wallet is oracle in cached PoCConfig."""
        self._oracle_verify_rpc_unreachable = False
        try:
            cfg = self.get_poc_config(bypass_cache=True)
            oracle = cfg.get("oracle_address")
            ours = self.wallet.get_address()
            ok = oracle == ours
            if ok:
                logger.info("Oracle authorized", our_address=ours, oracle_address=oracle)
            else:
                logger.error(
                    "Oracle NOT authorized",
                    our_address=ours,
                    configured_oracle=oracle,
                )
            return ok
        except requests.exceptions.RequestException as e:
            self._oracle_verify_rpc_unreachable = True
            logger.warning(
                "MySocial RPC unreachable; cannot verify oracle address",
                rpc_url=self.rpc_url,
                error=str(e),
            )
            return False
        except Exception as e:
            logger.warning(
                "Oracle authorization check failed (RPC error or unexpected response)",
                rpc_url=self.rpc_url,
                error=str(e),
            )
            return False


def init_myso_client() -> Optional[MySocialClient]:
    """Initialize MySocial client when `MYSO_INTEGRATION_ENABLED` is truthy."""

    from app.services.poc_utils import mys_integration_enabled_from_env, poc_strict_oracle_from_env

    if not mys_integration_enabled_from_env():
        logger.info("MySocial blockchain integration disabled")
        return None

    required = ["MYSO_POC_PACKAGE_ID", "MYSO_POC_CONFIG_ID", "MYSO_POC_REGISTRY_ID", "MYSO_POC_VAULT_DIRECTORY_ID"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        logger.error("PoC blockchain disabled — missing env", missing=missing)
        return None

    try:
        client = MySocialClient()
        if client.verify_oracle_authorization():
            logger.info("MySocial oracle authorized")
        else:
            if client._oracle_verify_rpc_unreachable:
                if poc_strict_oracle_from_env():
                    logger.error(
                        "PoC strict oracle: RPC unreachable — cannot verify wallet matches PoCConfig.oracle_address",
                    )
                    return None
                logger.warning(
                    "MySocial RPC unreachable; on-chain PoC calls will fail until MYSOCIAL_RPC_URL is reachable",
                    rpc_url=client.rpc_url,
                )
            elif poc_strict_oracle_from_env():
                logger.error(
                    "PoC strict oracle: wallet does not match PoCConfig.oracle_address — disabling client",
                )
                return None
            else:
                logger.warning("Oracle may not match PoCConfig oracle_address")

        return client
    except Exception as e:
        logger.error("Failed to initialize MySocial client", error=str(e))
        return None

