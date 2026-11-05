"""Oracle worker — processes analyze and claim jobs."""

from __future__ import annotations

import asyncio
import os
import time

import structlog

from app.chain.rpc_client import TransactionSubmitter
from app.db.discovery_repository import ProvenanceHitRepository
from app.db.oracle_repository import (
    AttestationRepository,
    ChainPostRepository,
    CompositionAnalysisRepository,
    ConfigCacheRepository,
    DetectedRelationshipRepository,
    DerivativeEdgeRepository,
    FingerprintObservationRepository,
    JobRepository,
    MediaAssetRepository,
    MediaAssetUsageRepository,
    MediaAssetRightsBundleRepository,
    PendingDerivativeAssetRepository,
    PostEnforcementSnapshotRepository,
)
from app.discovery.store import DiscoveryStore
from app.network_config import get_settings, load_network_profile
from app.services.analysis.media_fetcher import MediaNotReadyError
from app.services.analysis.pipeline import AnalysisService
from app.services.decision_engine import DecisionEngine
from app.services.dripdrop_video_client import extract_asset_id_from_hls, get_dripdrop_video_client
from app.services.events import event_bus
from app.services.proof_bundle import ProofBundleService
from app.services.username_beneficiary import UsernameBeneficiaryService
from app.services.vault_lifecycle import VaultLifecycleService
from app.services.poc_submission import build_composition_submission
from app.services.composition_submission import (
    AssetVersionInput,
    USAGE_SOCIAL_POST,
    build_analyze_post_composition_move_call,
)
from app.services.media_asset_submission import (
    ASSET_KIND_SOUND_RECORDING,
    ASSET_KIND_VISUAL_WORK,
    ORIGINALITY_DERIVATIVE,
    ORIGINALITY_ORIGINAL,
    MediaResolutionResult,
    USAGE_MUSIC_SOUNDTRACK,
    build_finalize_media_asset_move_call,
    build_submit_media_resolution_move_call,
    audio_rights_from_intent,
    usage_permitted,
    verified_visual_registration_rights,
)
from app.services.source_access import fetch_audio_source
from app.services.standalone_audio_resolution import (
    fetch_and_analyze_standalone_audio,
    requires_relationship_review,
)
from app.services.derivative_graph_submission import (
    ParentEdgeInput,
    PendingAssetInput,
    PROPOSAL_STATUS_PENDING,
    build_derivative_finalize_ptb,
    build_materialize_initial_resolved_policy_move_call,
    build_original_finalize_ptb,
    build_propose_detected_relationship_move_call,
)
from app.services.post_enforcement_submission import (
    MEDIA_COMPONENT_AUDIO,
    MEDIA_COMPONENT_IMAGE,
    MEDIA_COMPONENT_VIDEO,
    EmbeddedAssetBinding,
    audio_edit_recipe_commitment,
    build_record_embedded_bindings_move_call,
    build_refresh_post_usage_decisions_ptb,
    build_submit_candidate_revenue_manifest_move_call,
    deterministic_binding_id,
)
from app.services.media_asset_rights_governance_submission import (
    build_finalize_media_asset_rights_governance_move_call,
    build_implement_media_asset_rights_move_call,
    build_implement_media_asset_rights_to_beneficiary_move_call,
    build_implement_media_asset_rights_to_offchain_identity_move_call,
    build_implement_media_asset_rights_to_onchain_username_move_call,
    build_implement_media_asset_rights_to_profile_username_move_call,
)
from app.chain.bcs_media_asset_claims import (
    decode_claims_vector_bcs,
    decode_usage_grants_vector_bcs,
)
from app.services.rights_resolution import (
    TARGET_KIND_OFFCHAIN,
    TARGET_KIND_ONCHAIN_USERNAME,
    TARGET_KIND_VAULT,
    TARGET_KIND_WALLET,
    commitment_for_resolution,
    target_from_bundle_row,
)
from app.services.indexer_graphql_client import (
    PROPOSAL_STATUS_APPROVED,
    PROPOSAL_STATUS_COMMUNITY_VOTING,
    PROPOSAL_STATUS_IMPLEMENTED,
    PROPOSAL_STATUS_REJECTED,
    fetch_media_asset_usages,
    fetch_poc_governance_proposals,
    fetch_post_playback_policy,
    is_media_asset_rights_proposal,
)
from app.services.poc_utils import (
    DERIVATIVE_TARGET_WALLET,
    truncate_evidence_urls,
    truncate_reasoning,
)

logger = structlog.get_logger()

PENDING_FIRST_SIMILARITY_BPS = int(os.getenv("POC_PENDING_SIMILARITY_BPS", "8500"))

POC_GOV_POLL_INTERVAL_SECS = int(os.getenv("POC_GOVERNANCE_POLL_INTERVAL_SECS", "30"))
POC_GOV_IMPLEMENT_ENABLED = os.getenv("POC_GOVERNANCE_IMPLEMENT_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
)
POC_POST_REFRESH_AFTER_RIGHTS = os.getenv(
    "POC_POST_REFRESH_AFTER_RIGHTS_IMPLEMENT", "true"
).lower() in ("1", "true", "yes")


def _pending_first_enabled() -> bool:
    return os.getenv("POC_PENDING_FIRST_ENABLED", "").lower() in ("1", "true", "yes")


def _is_media_not_ready(exc: BaseException) -> bool:
    if isinstance(exc, MediaNotReadyError):
        return True
    # httpx errors that bubble from nested callers without MediaNotReadyError wrap
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in {404, 408, 425, 429, 503}


class OracleWorker:
    def __init__(self, network: str) -> None:
        self.network = network
        self.settings = get_settings()
        self.profile = load_network_profile(network)
        self.jobs = JobRepository()
        self.posts = ChainPostRepository()
        self.attestations = AttestationRepository()
        self.config_cache = ConfigCacheRepository()
        self.analysis = AnalysisService()
        self.decisions = DecisionEngine()
        self.proofs = ProofBundleService(network)
        self.beneficiaries = UsernameBeneficiaryService(network)
        self.vault_lifecycle = VaultLifecycleService(network)
        self.submitter = TransactionSubmitter(network)
        self.media_assets = MediaAssetRepository()
        self.fingerprint_obs = FingerprintObservationRepository()
        self.asset_usages = MediaAssetUsageRepository()
        self.composition_records = CompositionAnalysisRepository()
        self.pending_assets = PendingDerivativeAssetRepository()
        self.derivative_edges = DerivativeEdgeRepository()
        self.detected_relationships = DetectedRelationshipRepository()
        self.post_enforcement = PostEnforcementSnapshotRepository()
        self.rights_bundles = MediaAssetRightsBundleRepository()

    async def run_forever(self) -> None:
        logger.info("Oracle worker started", network=self.network)
        sem = asyncio.Semaphore(self.settings.oracle_worker_concurrency)
        if POC_GOV_IMPLEMENT_ENABLED:
            asyncio.create_task(self._poll_media_asset_rights_governance())
        while True:
            job = self.jobs.claim_next(self.network)
            if not job:
                await asyncio.sleep(1.0)
                continue
            asyncio.create_task(self._run_job(job, sem))

    async def _run_job(self, job: dict, sem: asyncio.Semaphore) -> None:
        async with sem:
            job_id = str(job["id"])
            job_type = job.get("job_type") or "analyze_post"
            attempts = int(job.get("attempts") or 1)
            max_retries = int(getattr(self.settings, "oracle_max_retries", 5) or 5)
            job_result = None
            try:
                if job_type == "claim_beneficiary":
                    job_result = await self._process_claim(job)
                elif job_type == "propose_beneficiary_transfer":
                    job_result = await self._process_beneficiary_transfer_proposal(job)
                elif job_type == "resolve_media_asset":
                    await self._process_resolve_media_asset(job)
                elif job_type == "resolve_post_media":
                    await self._process_resolve_post_media(job)
                elif job_type == "finalize_derivative":
                    await self._process_finalize_derivative(job)
                elif job_type == "finalize_original":
                    await self._process_finalize_original(job)
                elif job_type == "materialize_policy":
                    await self._process_materialize_policy(job)
                elif job_type == "refresh_post_usage_decisions":
                    await self._process_refresh_post_usage_decisions(job)
                elif job_type == "analyze_composition":
                    await self._process_analyze_composition(job)
                elif job_type == "poc_gov_finalize_rights":
                    await self._process_poc_gov_finalize_rights(job)
                elif job_type == "poc_gov_implement_rights":
                    await self._process_poc_gov_implement_rights(job)
                else:
                    await self._process_analyze(job)
                self.jobs.complete(job_id, "completed", result=job_result)
            except Exception as exc:
                if job_type not in {"claim_beneficiary", "resolve_media_asset", "resolve_post_media"} and _is_media_not_ready(exc) and attempts < max_retries:
                    # Exponential backoff: 15s, 30s, 60s, … capped at 120s
                    delay = min(120, 15 * (2 ** max(0, attempts - 1)))
                    logger.warning(
                        "Oracle media not ready; requeueing",
                        job_id=job_id,
                        post_id=job.get("post_id"),
                        attempts=attempts,
                        max_retries=max_retries,
                        delay_seconds=delay,
                        error=str(exc),
                    )
                    self.jobs.requeue(job_id, error=str(exc), delay_seconds=delay)
                    post_id = job.get("post_id")
                    if post_id:
                        self.posts.update_status(
                            self.network,
                            post_id,
                            analysis_status="pending_media",
                        )
                    return

                logger.exception("Oracle job failed", job_id=job_id, error=str(exc))
                self.jobs.complete(job_id, "failed", str(exc))
                await event_bus.publish(
                    "post.attestation.failed",
                    {
                        "network": self.network,
                        "post_id": job.get("post_id"),
                        "error": str(exc),
                    },
                )

    async def _process_analyze(self, job: dict) -> None:
        post_id = job["post_id"]
        media_url = job.get("media_url") or ""
        media_index = int(job.get("media_index") or 0)
        media_type = int(job.get("media_type") or 1)

        await event_bus.publish(
            "post.analysis.started",
            {"network": self.network, "post_id": post_id, "job_id": str(job["id"])},
        )
        self.posts.update_status(self.network, post_id, analysis_status="analyzing")

        post = self.posts.get(self.network, post_id) or {"post_id": post_id}
        metadata = post.get("metadata") or {}
        if isinstance(metadata, str):
            import json

            try:
                metadata = json.loads(metadata)
            except Exception:
                metadata = {}
        post_tx_digest = (
            post.get("tx_digest")
            or metadata.get("tx_digest")
            or metadata.get("transaction_digest")
            or ""
        )
        event_idx = int(metadata.get("event_sequence") or metadata.get("event_idx") or 0)
        event_sequence = event_idx * 1000 + media_index
        creator = post.get("creator_address") or ""

        analysis = await self.analysis.analyze(
            self.network,
            post_id,
            media_url,
            media_index,
            media_type,
            creator_wallet_address=creator or None,
            transaction_digest=str(post_tx_digest) or None,
            event_sequence=event_sequence,
            platform_id=post.get("platform_id") or metadata.get("platform_id"),
        )
        submission = self.decisions.build_submission(post, analysis)

        if submission.needs_review:
            self.posts.update_status(self.network, post_id, analysis_status="needs_review")
            return

        original_creator = submission.original_creator
        vault_provisioned = False
        if submission.off_network and submission.identity_hash and submission.original_creator is None:
            if submission.derivative_redirection_target == 1:
                original_creator, vault_provisioned = await self.vault_lifecycle.ensure_off_network_vault(
                    post_id=post_id,
                    identity_hash=submission.identity_hash,
                    username=analysis.matched_x_handle,
                    discovery_asset_id=analysis.discovery_asset_id,
                )
                if vault_provisioned and analysis.discovery_asset_id:
                    ProvenanceHitRepository().record(
                        network=self.network,
                        post_id=post_id,
                        query_media_id=analysis.media_id,
                        discovery_asset_id=analysis.discovery_asset_id,
                        creator_candidate_id=None,
                        similarity_score=analysis.highest_similarity_u64 / 100.0,
                        match_type="discovered",
                        work_confidence=analysis.work_confidence,
                        creator_confidence=analysis.creator_confidence,
                        decision="redirect_escrow",
                        vault_provisioned=True,
                        vault_identity_hash=submission.identity_hash,
                    )
                    DiscoveryStore().transition_asset(analysis.discovery_asset_id, "match_detected")
                    await self.vault_lifecycle.mark_claimable(
                        submission.identity_hash,
                        analysis.discovery_asset_id,
                    )

        cfg = self.submitter.get_poc_config()
        self.config_cache.set(self.network, cfg)
        reasoning = truncate_reasoning(
            submission.reasoning,
            int(cfg.get("max_reasoning_length") or 5000),
        )
        evidence = truncate_evidence_urls(submission.evidence_urls, int(cfg.get("max_evidence_urls") or 10))

        bundle = self.proofs.build_bundle(post, analysis, submission)
        proof_uri = self.proofs.store(bundle)
        if evidence is None:
            evidence = [proof_uri]
        else:
            evidence = list(evidence) + [proof_uri]

        tx = self.submitter.submit_analysis(
            post_id=post_id,
            media_type=submission.media_type,
            highest_similarity_score=submission.highest_similarity_score,
            original_creator=original_creator,
            derivative_redirection_target=submission.derivative_redirection_target,
            embedded_audio_only_derivative=submission.embedded_audio_only_derivative,
            apply_explicit_outcome=submission.apply_explicit_outcome,
            explicit_poc_outcome=submission.explicit_poc_outcome,
            reasoning=reasoning,
            evidence_urls=evidence,
        )
        tx_digest = tx.get("tx_hash")
        await event_bus.publish(
            "post.attestation.submitted",
            {"network": self.network, "post_id": post_id, "tx_digest": tx_digest},
        )

        self.attestations.insert(
            {
                "network": self.network,
                "post_id": post_id,
                "tx_digest": tx_digest,
                "media_type": submission.media_type,
                "highest_similarity_score": submission.highest_similarity_score,
                "original_creator": original_creator,
                "derivative_redirection_target": submission.derivative_redirection_target,
                "reasoning": reasoning,
                "evidence_urls": evidence,
                "status": "submitted",
            }
        )
        if tx_digest and not tx.get("mock"):
            self.attestations.mark_confirmed(self.network, str(tx_digest))
        self.posts.update_status(
            self.network,
            post_id,
            analysis_status="attested",
            highest_similarity_score=submission.highest_similarity_score,
            proof_bundle_uri=proof_uri,
            tx_digest=tx_digest,
        )
        await event_bus.publish(
            "post.attestation.confirmed",
            {
                "network": self.network,
                "post_id": post_id,
                "tx_digest": tx_digest,
                "poc_outcome": None,
            },
        )

        # Notify dripdrop-backend to start private-source retention (HLS posts only).
        try:
            dd = get_dripdrop_video_client()
            asset_id = extract_asset_id_from_hls(media_url, dd.media_host)
            if asset_id and dd.enabled and creator and post_tx_digest:
                await dd.analysis_complete(
                    asset_id=asset_id,
                    post_object_id=post_id,
                    creator_wallet_address=creator,
                    hls_url=media_url,
                    transaction_digest=str(post_tx_digest),
                    event_sequence=event_sequence,
                    status="succeeded",
                )
        except Exception as exc:
            logger.warning(
                "analysis-complete notify failed",
                post_id=post_id,
                error=str(exc),
            )

    async def _process_resolve_post_media(self, job: dict) -> None:
        """Post-first path: submit resolution for each unresolved clip, then compose."""
        payload = job.get("payload") or {}
        if isinstance(payload, str):
            import json

            payload = json.loads(payload)

        post_id = str(payload.get("post_id") or job.get("post_id") or "")
        submitter = str(payload.get("submitter") or "")
        clips = payload.get("clips")
        if not isinstance(clips, list) or not clips:
            fp_hex = str(payload.get("observed_fingerprint_commitment") or "")
            content_hex = str(payload.get("content_commitment") or "")
            if fp_hex and content_hex:
                clips = [
                    {
                        "content_commitment": content_hex,
                        "observed_fingerprint_commitment": fp_hex,
                        "media_type": int(payload.get("media_type") or 2),
                    }
                ]
            else:
                clips = []

        if not post_id or not clips:
            raise ValueError("resolve_post_media requires post_id and fingerprint/content commitments")

        resolved_ids: list[str] = []
        from app.chain.ptb_builder import extract_created_object_id, extract_event_field

        for clip in clips:
            if not isinstance(clip, dict):
                continue
            existing_clip = str(clip.get("media_asset_id") or "")
            if existing_clip:
                if existing_clip not in resolved_ids:
                    resolved_ids.append(existing_clip)
                continue
            content_hex = str(clip.get("content_commitment") or payload.get("content_commitment") or "")
            fp_hex = str(
                clip.get("observed_fingerprint_commitment")
                or payload.get("observed_fingerprint_commitment")
                or ""
            )
            media_type = int(clip.get("media_type") or payload.get("media_type") or 2)
            if not fp_hex or not content_hex:
                raise ValueError("resolve_post_media requires post_id and fingerprint/content commitments")

            submit_mc = build_submit_media_resolution_move_call(
                content_commitment=bytes.fromhex(content_hex),
                observed_fingerprint_commitment=bytes.fromhex(fp_hex),
                media_type=media_type,
            )
            submit_tx = self.submitter.submit_move_call(submit_mc)
            request_id = extract_event_field(submit_tx, "MediaResolutionRequestedEvent", "request_id")
            if not request_id:
                request_id = extract_created_object_id(submit_tx)
            if not request_id:
                raise ValueError("resolve_post_media: could not parse MediaResolutionRequestedEvent.request_id")

            clip_payload = dict(payload)
            clip_payload.update(clip)
            resolved_asset_id = await self._finalize_media_resolution(
                request_id=request_id,
                fp_hex=fp_hex,
                content_hex=content_hex,
                media_type=media_type,
                submitter=submitter,
                payload=clip_payload,
            )
            if not resolved_asset_id:
                raise ValueError(f"resolve_post_media: finalize produced no asset id for post {post_id}")
            if resolved_asset_id not in resolved_ids:
                resolved_ids.append(str(resolved_asset_id))

        existing = [str(a) for a in (payload.get("existing_media_asset_ids") or []) if a]
        media_asset_ids = existing + [a for a in resolved_ids if a not in existing]
        if not media_asset_ids:
            raise ValueError(f"resolve_post_media: finalize produced no asset id for post {post_id}")

        job_id = self.jobs.enqueue(
            self.network,
            post_id,
            job_type="analyze_composition",
            payload={
                "media_asset_ids": media_asset_ids,
                "spt_id": payload.get("spt_id"),
                "tx_digest": payload.get("tx_digest"),
                "platform_id": payload.get("platform_id"),
            },
        )
        logger.info(
            "Enqueued analyze_composition after resolve_post_media",
            network=self.network,
            post_id=post_id,
            asset_id=media_asset_ids[0],
            job_id=job_id,
        )

    async def _finalize_media_resolution(
        self,
        *,
        request_id: str,
        fp_hex: str,
        content_hex: str,
        media_type: int,
        submitter: str,
        payload: dict,
    ) -> str:
        """Run finalize_media_asset for a resolution request; return canonical MediaAsset.id."""
        if _pending_first_enabled() and media_type != 3:
            await self._process_resolve_media_asset_pending_first(
                request_id=request_id,
                fp_hex=fp_hex,
                content_hex=content_hex,
                media_type=media_type,
                submitter=submitter,
                payload=payload,
            )
            pending = self.pending_assets.find_by_request(self.network, request_id) or {}
            child_id = pending.get("child_asset_id")
            if child_id:
                return str(child_id)
            pending_id = pending.get("pending_id")
            if pending_id:
                row = self.pending_assets.get(self.network, str(pending_id)) or {}
                if row.get("child_asset_id"):
                    return str(row["child_asset_id"])
            asset = self.media_assets.find_by_fingerprint(self.network, fp_hex)
            if asset:
                return str(asset["asset_id"])
            return str(request_id)

        existing_asset_id = self.fingerprint_obs.find_asset_for_fingerprint(self.network, fp_hex)
        if not existing_asset_id:
            existing = self.media_assets.find_by_fingerprint(self.network, fp_hex)
            existing_asset_id = str(existing["asset_id"]) if existing else None

        link_existing = existing_asset_id
        originality = ORIGINALITY_DERIVATIVE if link_existing else ORIGINALITY_ORIGINAL
        lineage = link_existing if link_existing else None

        if link_existing:
            existing_rights = self.media_assets.get(self.network, str(link_existing)) or {}
            rights_payload = existing_rights.get("rights_json") or {}
            if not rights_payload.get("claims") or not rights_payload.get("usage_grants"):
                raise ValueError("dedup target has no verified rights snapshot")
        elif media_type == 3:
            rights_payload = audio_rights_from_intent(
                intent=dict(payload.get("rights_intent") or {}),
                wallet_signature=str(payload.get("wallet_signature") or ""),
                submitter=submitter,
                content_commitment_hex=content_hex,
            )
        else:
            rights_payload = verified_visual_registration_rights(
                submitter=submitter,
                content_commitment_hex=content_hex,
                fingerprint_commitment_hex=fp_hex,
            )

        resolution = MediaResolutionResult(
            request_id=request_id,
            content_commitment=bytes.fromhex(content_hex) if content_hex else b"",
            observed_fingerprint_commitment=bytes.fromhex(fp_hex) if fp_hex else b"",
            media_type=media_type,
            submitter=submitter,
            link_to_existing_id=link_existing,
            originality_status=originality,
            lineage_parent_id=lineage,
            asset_kind=(ASSET_KIND_SOUND_RECORDING if media_type == 3 else ASSET_KIND_VISUAL_WORK),
            claims=list(rights_payload.get("claims") or []),
            usage_grants=list(rights_payload.get("usage_grants") or []),
            dedup_matched=bool(link_existing),
        )
        move_call = build_finalize_media_asset_move_call(resolution)
        tx = self.submitter.submit_finalize_media_asset(move_call)
        tx_digest = tx.get("tx_hash")

        from app.chain.ptb_builder import extract_created_object_id, extract_media_asset_resolved_id

        resolved_asset_id = extract_media_asset_resolved_id(tx)
        if not resolved_asset_id:
            resolved_asset_id = extract_created_object_id(tx)
        if not resolved_asset_id:
            resolved_asset_id = link_existing or request_id

        if link_existing:
            self.fingerprint_obs.record(
                self.network,
                {
                    "fingerprint_commitment": fp_hex,
                    "content_commitment": content_hex,
                    "media_asset_id": resolved_asset_id,
                    "request_id": request_id,
                    "media_type": media_type,
                    "submitter": submitter,
                },
            )
            await event_bus.publish(
                "media_asset.resolved",
                {
                    "network": self.network,
                    "request_id": request_id,
                    "asset_id": resolved_asset_id,
                    "linked_existing": True,
                    "tx_digest": tx_digest,
                },
            )
            return str(resolved_asset_id)

        # Original works must expose a resolved policy before product use is allowed.
        self.submitter.submit_move_call(
            build_materialize_initial_resolved_policy_move_call(str(resolved_asset_id))
        )
        claim_creators = [
            c["claimant"]
            for c in rights_payload.get("claims", [])
            if int(c.get("claim_type", 0)) == 1
        ]
        rights_controllers = [
            c["claimant"]
            for c in rights_payload.get("claims", [])
            if int(c.get("claim_type", 0)) in (2, 3)
        ]
        self.fingerprint_obs.record(
            self.network,
            {
                "fingerprint_commitment": fp_hex,
                "content_commitment": content_hex,
                "media_asset_id": resolved_asset_id,
                "request_id": request_id,
                "media_type": media_type,
                "submitter": submitter,
            },
        )
        self.media_assets.upsert(
            self.network,
            {
                "asset_id": resolved_asset_id,
                "content_commitment": content_hex,
                "fingerprint_commitment": fp_hex,
                "media_type": media_type,
                "originality_status": originality,
                "lineage_parent_id": lineage,
                "creators": claim_creators or ([submitter] if submitter else []),
                "rights_controllers": rights_controllers,
                "rights_json": rights_payload,
                "linked_existing": bool(link_existing),
                "resolve_tx_digest": tx_digest,
                "request_id": request_id,
            },
        )
        await event_bus.publish(
            "media_asset.resolved",
            {
                "network": self.network,
                "request_id": request_id,
                "asset_id": resolved_asset_id,
                "linked_existing": bool(link_existing),
                "tx_digest": tx_digest,
            },
        )
        return str(resolved_asset_id)

    async def _process_resolve_media_asset(self, job: dict) -> None:
        payload = job.get("payload") or {}
        if isinstance(payload, str):
            import json

            payload = json.loads(payload)

        request_id = str(payload.get("request_id") or job.get("post_id") or "")
        fp_hex = str(payload.get("observed_fingerprint_commitment") or "")
        content_hex = str(payload.get("content_commitment") or "")
        media_type = int(payload.get("media_type") or 1)
        submitter = str(payload.get("submitter") or "")

        if not request_id or not fp_hex:
            raise ValueError("resolve_media_asset requires request_id and observed_fingerprint_commitment")

        if media_type == 3:
            public_url = str(
                payload.get("source_url")
                or payload.get("sourceUrl")
                or (payload.get("media_urls") or [None])[0]
                or ""
            ) or None
            source_access = await fetch_audio_source(
                content_commitment=f"0x{content_hex.removeprefix('0x')}",
                resolution_request_id=request_id,
                platform_id=payload.get("platform_id"),
                public_url=public_url,
            )
            source_url = str(source_access.get("sourceUrl") or "")
            if not source_url:
                raise MediaNotReadyError("audio-source-lease", message="audio source lease missing URL")
            analysis = await fetch_and_analyze_standalone_audio(
                source_url=source_url,
                content_commitment_hex=content_hex,
                observed_fingerprint_commitment_hex=fp_hex,
            )
            known_exact = self.fingerprint_obs.find_asset_for_fingerprint(self.network, fp_hex)
            if not known_exact and requires_relationship_review(analysis):
                raise ValueError("standalone audio resembles existing work and requires rights review")
            owner = str(source_access.get("ownerWalletAddress") or "")
            if owner and owner != submitter:
                raise ValueError("audio source owner does not match chain submitter")
            if source_access.get("rightsIntent"):
                payload["rights_intent"] = source_access.get("rightsIntent") or {}
            if source_access.get("walletSignature"):
                payload["wallet_signature"] = source_access.get("walletSignature") or ""

        await self._finalize_media_resolution(
            request_id=request_id,
            fp_hex=fp_hex,
            content_hex=content_hex,
            media_type=media_type,
            submitter=submitter,
            payload=payload,
        )

    async def _process_resolve_media_asset_pending_first(
        self,
        *,
        request_id: str,
        fp_hex: str,
        content_hex: str,
        media_type: int,
        submitter: str,
        payload: dict,
    ) -> None:
        content_bytes = bytes.fromhex(content_hex) if content_hex else b""
        pending_input = PendingAssetInput(
            content_commitment=content_bytes,
            media_type=media_type,
        )
        recipe = build_original_finalize_ptb(pending_input)
        tx = self.submitter.submit_ptb(recipe.to_dict()["steps"])
        tx_digest = tx.get("tx_hash")
        created = tx.get("created_objects") or {}
        pending_id = created.get(0) or request_id
        child_asset_id = created.get(len(recipe.steps) - 1)

        from app.chain.ptb_builder import extract_created_object_id, extract_media_asset_resolved_id

        resolved_from_event = extract_media_asset_resolved_id(tx) or extract_created_object_id(tx)
        if resolved_from_event:
            child_asset_id = resolved_from_event

        canonical_parent = self.fingerprint_obs.find_asset_for_fingerprint(self.network, fp_hex)
        if not canonical_parent:
            existing = self.media_assets.find_by_fingerprint(self.network, fp_hex)
            canonical_parent = str(existing["asset_id"]) if existing else None

        if canonical_parent:
            similarity_bps = int(payload.get("similarity_bps") or PENDING_FIRST_SIMILARITY_BPS)
            config_id = os.getenv("MYSO_POC_CONFIG_ID", "")
            if config_id:
                propose_mc = build_propose_detected_relationship_move_call(
                    config_id=config_id,
                    accused_pending_id=pending_id,
                    original_asset_id=canonical_parent,
                    similarity_bps=similarity_bps,
                    evidence_commitment=content_bytes or None,
                )
                propose_tx = self.submitter.submit_move_call(propose_mc)
                proposal_id = str(propose_tx.get("tx_hash") or pending_id)
                self.detected_relationships.upsert(
                    self.network,
                    {
                        "proposal_id": proposal_id,
                        "accused_pending_id": pending_id,
                        "original_asset_id": canonical_parent,
                        "similarity_bps": similarity_bps,
                        "status": PROPOSAL_STATUS_PENDING,
                        "tx_digest": propose_tx.get("tx_hash"),
                    },
                )

        self.pending_assets.upsert(
            self.network,
            {
                "pending_id": pending_id,
                "request_id": request_id,
                "content_commitment": content_hex,
                "fingerprint_commitment": fp_hex,
                "media_type": media_type,
                "creator": submitter,
                "status": "finalized" if child_asset_id else "pending",
                "finalize_tx_digest": tx_digest,
                "child_asset_id": child_asset_id,
            },
        )

        resolved_asset_id = child_asset_id or pending_id
        rights_payload = verified_visual_registration_rights(
            submitter=submitter,
            content_commitment_hex=content_hex,
            fingerprint_commitment_hex=fp_hex,
        )
        self.fingerprint_obs.record(
            self.network,
            {
                "fingerprint_commitment": fp_hex,
                "content_commitment": content_hex,
                "media_asset_id": resolved_asset_id,
                "request_id": request_id,
                "media_type": media_type,
                "submitter": submitter,
            },
        )
        self.media_assets.upsert(
            self.network,
            {
                "asset_id": resolved_asset_id,
                "content_commitment": content_hex,
                "fingerprint_commitment": fp_hex,
                "media_type": media_type,
                "originality_status": ORIGINALITY_ORIGINAL,
                "pending_id": pending_id,
                "rights_json": rights_payload,
                "resolve_tx_digest": tx_digest,
                "request_id": request_id,
            },
        )
        await event_bus.publish(
            "media_asset.resolved",
            {
                "network": self.network,
                "request_id": request_id,
                "pending_id": pending_id,
                "asset_id": resolved_asset_id,
                "tx_digest": tx_digest,
                "pending_first": True,
            },
        )

    async def _process_finalize_derivative(self, job: dict) -> None:
        payload = job.get("payload") or {}
        if isinstance(payload, str):
            import json

            payload = json.loads(payload)
        pending_input = PendingAssetInput(
            content_commitment=bytes.fromhex(str(payload.get("content_commitment") or "")),
            media_type=int(payload.get("media_type") or 1),
        )
        parents = [
            ParentEdgeInput(
                parent_asset_id=str(p["parent_asset_id"]),
                license_instance_id=str(p["license_instance_id"]),
                template_version_id=str(p["template_version_id"]),
                relationship_type=int(p.get("relationship_type") or 1),
            )
            for p in (payload.get("parents") or [])
        ]
        recipe = build_derivative_finalize_ptb(pending_input, parents)
        tx = self.submitter.submit_ptb(recipe.to_dict()["steps"])
        child_id = (tx.get("created_objects") or {}).get(len(recipe.steps) - 1)
        if child_id and parents:
            for edge in parents:
                self.derivative_edges.record(
                    self.network,
                    {
                        "parent_asset_id": edge.parent_asset_id,
                        "child_asset_id": child_id,
                        "relationship_type": edge.relationship_type,
                        "license_instance_id": edge.license_instance_id,
                        "template_version_id": edge.template_version_id,
                        "tx_digest": tx.get("tx_hash"),
                    },
                )

    async def _process_finalize_original(self, job: dict) -> None:
        payload = job.get("payload") or {}
        if isinstance(payload, str):
            import json

            payload = json.loads(payload)
        pending_input = PendingAssetInput(
            content_commitment=bytes.fromhex(str(payload.get("content_commitment") or "")),
            media_type=int(payload.get("media_type") or 1),
        )
        recipe = build_original_finalize_ptb(pending_input)
        self.submitter.submit_ptb(recipe.to_dict()["steps"])

    async def _process_materialize_policy(self, job: dict) -> None:
        payload = job.get("payload") or {}
        if isinstance(payload, str):
            import json

            payload = json.loads(payload)
        asset_id = str(payload.get("asset_id") or "")
        if not asset_id:
            raise ValueError("materialize_policy requires asset_id")
        mc = build_materialize_initial_resolved_policy_move_call(asset_id)
        self.submitter.submit_move_call(mc)

    async def _process_refresh_post_usage_decisions(self, job: dict) -> None:
        payload = job.get("payload") or {}
        if isinstance(payload, str):
            import json

            payload = json.loads(payload)
        post_id = str(payload.get("post_id") or job.get("post_id") or "")
        bindings_payload = payload.get("bindings") or []
        binding_ids = [int(b) for b in (payload.get("binding_ids") or [])]
        oracle_address = os.getenv("MYSO_POC_ORACLE_ADDRESS", "")
        if not oracle_address and self.submitter.client:
            oracle_address = self.submitter.client.wallet.get_address()
        if not post_id or (not bindings_payload and not binding_ids) or not oracle_address:
            raise ValueError(
                "refresh_post_usage_decisions requires post_id, bindings or binding_ids, oracle"
            )
        if bindings_payload:
            bindings = [
                EmbeddedAssetBinding(
                    binding_id=int(b.get("binding_id") or b.get("bindingId") or 0),
                    source_asset_id=str(b.get("source_asset_id") or b.get("sourceAssetId") or ""),
                    usage_class=int(b.get("usage_class") or b.get("usageClass") or 1),
                    stem=int(b.get("stem") or 0),
                    media_component=int(b.get("media_component") or b.get("mediaComponent") or 0),
                )
                for b in bindings_payload
            ]
        else:
            bindings = []
            for bid in binding_ids:
                source = str(payload.get("source_asset_id") or "")
                bindings.append(
                    EmbeddedAssetBinding(
                        binding_id=bid,
                        source_asset_id=source,
                        usage_class=1,
                        stem=0,
                        media_component=0,
                    )
                )
        steps = build_refresh_post_usage_decisions_ptb(
            oracle_address=oracle_address,
            post_id=post_id,
            bindings=bindings,
        )
        tx = self.submitter.submit_ptb(steps)
        gql = fetch_post_playback_policy(self.profile, post_id)
        if gql:
            self.post_enforcement.upsert(
                self.network,
                {
                    "post_id": post_id,
                    "bindings_json": gql.get("embeddedBindings") or [],
                    "usage_decisions_json": gql.get("usageDecisions") or [],
                    "usage_denials_json": gql.get("usageDenials") or [],
                    "playback_policy_json": gql.get("playbackPolicy") or {},
                    "tx_digest": tx.get("tx_hash"),
                },
            )

    async def _poll_media_asset_rights_governance(self) -> None:
        while True:
            try:
                await self._poll_media_asset_rights_governance_once()
            except Exception as exc:
                logger.exception(
                    "Media asset rights governance poll failed",
                    network=self.network,
                    error=str(exc),
                )
            await asyncio.sleep(max(5, POC_GOV_POLL_INTERVAL_SECS))

    async def _poll_media_asset_rights_governance_once(self) -> None:
        if not os.getenv("POC_GOVERNANCE_REGISTRY_ID", "").strip():
            return
        now_ms = int(time.time() * 1000)
        voting = fetch_poc_governance_proposals(
            self.profile, status=PROPOSAL_STATUS_COMMUNITY_VOTING, limit=20
        )
        for proposal in voting:
            if not is_media_asset_rights_proposal(proposal):
                continue
            voting_end = proposal.get("votingEndTime")
            if voting_end is None or int(voting_end) > now_ms:
                continue
            proposal_id = str(proposal["proposalId"])
            media_asset_id = str(proposal.get("referenceId") or "")
            if not media_asset_id:
                continue
            if self.jobs.has_active_job_for_proposal(
                self.network, "poc_gov_finalize_rights", proposal_id
            ):
                continue
            self.jobs.enqueue(
                self.network,
                media_asset_id,
                job_type="poc_gov_finalize_rights",
                payload={
                    "proposal_id": proposal_id,
                    "media_asset_id": media_asset_id,
                },
            )
            logger.info(
                "Enqueued media asset rights finalize job",
                proposal_id=proposal_id,
                media_asset_id=media_asset_id,
            )

        approved = fetch_poc_governance_proposals(
            self.profile, status=PROPOSAL_STATUS_APPROVED, limit=20
        )
        for proposal in approved:
            if not is_media_asset_rights_proposal(proposal):
                continue
            proposal_id = str(proposal["proposalId"])
            media_asset_id = str(proposal.get("referenceId") or "")
            if not media_asset_id:
                continue
            bundle = self.rights_bundles.get(proposal_id)
            if not bundle or str(bundle.get("status")) != "pending":
                continue
            if self.jobs.has_active_job_for_proposal(
                self.network, "poc_gov_implement_rights", proposal_id
            ):
                continue
            self.jobs.enqueue(
                self.network,
                media_asset_id,
                job_type="poc_gov_implement_rights",
                payload={
                    "proposal_id": proposal_id,
                    "media_asset_id": media_asset_id,
                    "reasoning": proposal.get("description") or "",
                },
            )
            logger.info(
                "Enqueued media asset rights implement job",
                proposal_id=proposal_id,
                media_asset_id=media_asset_id,
            )

        rejected = fetch_poc_governance_proposals(
            self.profile, status=PROPOSAL_STATUS_REJECTED, limit=20
        )
        for proposal in rejected:
            if not is_media_asset_rights_proposal(proposal):
                continue
            proposal_id = str(proposal["proposalId"])
            bundle = self.rights_bundles.get(proposal_id)
            if bundle and str(bundle.get("status")) == "pending":
                self.rights_bundles.mark_status(proposal_id, "rejected")

        implemented = fetch_poc_governance_proposals(
            self.profile, status=PROPOSAL_STATUS_IMPLEMENTED, limit=20
        )
        for proposal in implemented:
            if not is_media_asset_rights_proposal(proposal):
                continue
            proposal_id = str(proposal["proposalId"])
            bundle = self.rights_bundles.get(proposal_id)
            if bundle and str(bundle.get("status")) == "pending":
                self.rights_bundles.mark_status(proposal_id, "implemented")

    async def _process_poc_gov_finalize_rights(self, job: dict) -> None:
        payload = job.get("payload") or {}
        if isinstance(payload, str):
            import json

            payload = json.loads(payload)
        proposal_id = str(payload["proposal_id"])
        media_asset_id = str(payload["media_asset_id"])
        move_call = build_finalize_media_asset_rights_governance_move_call(
            proposal_id=proposal_id,
            media_asset_id=media_asset_id,
        )
        self.submitter.submit_move_call(move_call)

    def _rights_implement_move_call(
        self,
        *,
        proposal_id: str,
        media_asset_id: str,
        claims: list,
        usage_grants: list,
        reasoning: str,
        evidence_urls: list[str] | None,
        target: dict,
    ) -> dict:
        from app.chain.ptb_builder import extract_event_field
        from app.services.myso_client import _normalize_object_id
        from app.services.poc_chain_helpers import (
            fetch_profile_for_username,
            fetch_username_beneficiary_by_identity,
            fetch_username_beneficiary_by_username,
        )

        kind = int(target["target_kind"])
        profile = self.submitter.profile
        committed = _normalize_object_id(str(target["beneficiary_address"]))

        if kind == TARGET_KIND_WALLET:
            return build_implement_media_asset_rights_move_call(
                proposal_id=proposal_id,
                media_asset_id=media_asset_id,
                claims=claims,
                usage_grants=usage_grants,
                reasoning=reasoning,
                evidence_urls=evidence_urls,
            )

        if kind == TARGET_KIND_VAULT:
            vault_id = target.get("vault_id")
            if not vault_id:
                raise ValueError(f"kind 1 bundle {proposal_id} is missing target_vault_id")
            return build_implement_media_asset_rights_to_beneficiary_move_call(
                profile=profile,
                proposal_id=proposal_id,
                media_asset_id=media_asset_id,
                vault_id=str(vault_id),
                claims=claims,
                usage_grants=usage_grants,
                reasoning=reasoning,
            )

        if kind == TARGET_KIND_ONCHAIN_USERNAME:
            username = str(target.get("username") or "")
            existing = fetch_username_beneficiary_by_username(profile, username)
            if existing and existing.get("beneficiaryId"):
                found = _normalize_object_id(str(existing.get("beneficiaryAddress") or ""))
                if found != committed:
                    raise ValueError(
                        f"username beneficiary for {username} does not match the committed address"
                    )
                return build_implement_media_asset_rights_to_onchain_username_move_call(
                    profile=profile,
                    proposal_id=proposal_id,
                    media_asset_id=media_asset_id,
                    beneficiary_id=str(existing["beneficiaryId"]),
                    username=username,
                    claims=claims,
                    usage_grants=usage_grants,
                    reasoning=reasoning,
                )
            registered = fetch_profile_for_username(profile, username)
            if not registered:
                raise ValueError(
                    f"kind 2 username {username} has no beneficiary and no profile"
                )
            owner = _normalize_object_id(registered["owner"])
            if owner != committed:
                raise ValueError(
                    f"profile owner for {username} does not match the committed address"
                )
            return build_implement_media_asset_rights_to_profile_username_move_call(
                profile=profile,
                proposal_id=proposal_id,
                media_asset_id=media_asset_id,
                profile_id=registered["profile_id"],
                username=username,
                claims=claims,
                usage_grants=usage_grants,
                reasoning=reasoning,
            )

        if kind == TARGET_KIND_OFFCHAIN:
            username = str(target.get("username") or "")
            identity_hash = str(target.get("identity_hash") or "")
            source = int(target["identity_source"]) if target.get("identity_source") is not None else 1
            existing = fetch_username_beneficiary_by_identity(
                profile, identity_hash, identity_source=source
            )
            beneficiary_id = str(existing["beneficiaryId"]) if existing and existing.get("beneficiaryId") else None
            if beneficiary_id:
                found = _normalize_object_id(str(existing.get("beneficiaryAddress") or ""))
                if found != committed:
                    raise ValueError(
                        "identity beneficiary does not match the committed address"
                    )
            else:
                if not username.startswith("x_"):
                    raise RuntimeError(
                        "kind 3 identity is not indexed yet; retry implement after the existing vault is visible"
                    )
                created = self.submitter.create_username_beneficiary(
                    username=username,
                    identity_hash=identity_hash,
                    required_x_handle=username[2:],
                )
                beneficiary_id = extract_event_field(
                    created, "UsernameBeneficiaryProvisionedEvent", "beneficiary_id"
                )
                if not beneficiary_id:
                    raise RuntimeError(
                        "username beneficiary was created but its id was not in the transaction events; retry implement"
                    )
            return build_implement_media_asset_rights_to_offchain_identity_move_call(
                profile=profile,
                proposal_id=proposal_id,
                media_asset_id=media_asset_id,
                beneficiary_id=str(beneficiary_id),
                username=username,
                claims=claims,
                usage_grants=usage_grants,
                reasoning=reasoning,
            )

        raise ValueError(f"unknown rights target kind {kind}")

    async def _process_poc_gov_implement_rights(self, job: dict) -> None:
        import json

        payload = job.get("payload") or {}
        if isinstance(payload, str):
            payload = json.loads(payload)
        proposal_id = str(payload["proposal_id"])
        media_asset_id = str(payload["media_asset_id"])
        bundle = self.rights_bundles.get(proposal_id)
        if not bundle:
            raise ValueError(f"no rights bundle stored for proposal {proposal_id}")
        if str(bundle.get("status")) != "pending":
            logger.info(
                "Skipping implement; bundle already processed",
                proposal_id=proposal_id,
                status=bundle.get("status"),
            )
            return

        claims_bcs = bundle["claims_bcs"]
        grants_bcs = bundle["usage_grants_bcs"]
        if isinstance(claims_bcs, memoryview):
            claims_bcs = claims_bcs.tobytes()
        if isinstance(grants_bcs, memoryview):
            grants_bcs = grants_bcs.tobytes()
        claims = decode_claims_vector_bcs(bytes(claims_bcs))
        usage_grants = decode_usage_grants_vector_bcs(bytes(grants_bcs))
        commitment = bundle["claims_commitment"]
        if isinstance(commitment, memoryview):
            commitment = commitment.tobytes()
        target = target_from_bundle_row(bundle)
        expected = commitment_for_resolution(claims, usage_grants, target)
        if bytes(commitment) != expected:
            raise ValueError(
                f"rights resolution commitment mismatch for proposal {proposal_id}: "
                f"stored={bytes(commitment).hex()} computed={expected.hex()}"
            )

        reasoning = str(payload.get("reasoning") or "DAO approved media asset rights update")
        move_call = self._rights_implement_move_call(
            proposal_id=proposal_id,
            media_asset_id=media_asset_id,
            claims=claims,
            usage_grants=usage_grants,
            reasoning=reasoning,
            evidence_urls=payload.get("evidence_urls"),
            target=target,
        )
        self.submitter.submit_move_call(move_call)
        self.rights_bundles.mark_status(proposal_id, "implemented")

        if POC_POST_REFRESH_AFTER_RIGHTS:
            await self._enqueue_post_refresh_for_asset(media_asset_id)

    async def _enqueue_post_refresh_for_asset(self, media_asset_id: str) -> None:
        usages = fetch_media_asset_usages(self.profile, media_asset_id, limit=100)
        for usage in usages:
            if int(usage.get("containerType") or 0) != 1:
                continue
            post_id = str(usage.get("containerId") or "")
            if not post_id:
                continue
            post_data = fetch_post_playback_policy(self.profile, post_id)
            if not post_data:
                continue
            bindings = post_data.get("embeddedBindings") or []
            relevant = [
                b
                for b in bindings
                if str(b.get("sourceAssetId") or "") == media_asset_id
            ]
            if not relevant:
                continue
            self.jobs.enqueue(
                self.network,
                post_id,
                job_type="refresh_post_usage_decisions",
                payload={
                    "post_id": post_id,
                    "bindings": relevant,
                },
            )
            logger.info(
                "Enqueued post refresh after rights implement",
                post_id=post_id,
                media_asset_id=media_asset_id,
            )

    async def _process_analyze_composition(self, job: dict) -> None:
        post_id = job["post_id"]
        payload = job.get("payload") or {}
        if isinstance(payload, str):
            import json

            payload = json.loads(payload)

        media_asset_ids = payload.get("media_asset_ids") or []
        media_asset_roles = payload.get("media_asset_roles") or {}
        audio_edit_recipe = payload.get("audio_edit_recipe")
        if not media_asset_ids:
            post_row = self.posts.get(self.network, post_id) or {}
            media_asset_ids = post_row.get("media_asset_ids") or []
        if not media_asset_ids:
            raise ValueError("analyze_composition requires media_asset_ids")

        await event_bus.publish(
            "post.analysis.started",
            {"network": self.network, "post_id": post_id, "job_id": str(job["id"]), "mode": "composition"},
        )
        self.posts.update_status(self.network, post_id, analysis_status="analyzing")

        post = self.posts.get(self.network, post_id) or {"post_id": post_id}
        media_urls = post.get("media_urls") or []
        if isinstance(media_urls, str):
            media_urls = [media_urls]

        asset_inputs: list[AssetVersionInput] = []
        manifest_entries: list[AssetVersionInput] = []
        contains_derivatives = False
        contains_unresolved = False
        reasoning_parts: list[str] = []
        evidence: list[str] = []
        derivative_target = DERIVATIVE_TARGET_WALLET

        for idx, asset_id in enumerate(media_asset_ids):
            asset_id = str(asset_id)
            row = self.media_assets.get(self.network, asset_id)
            if not row:
                contains_unresolved = True
                reasoning_parts.append(f"asset {asset_id} unresolved")
                continue

            rights = row.get("rights_json") or {}
            media_type = int(row.get("media_type") or 1)
            role_value = media_asset_roles.get(asset_id) if isinstance(media_asset_roles, dict) else None
            role = (
                str(role_value.get("role") or "")
                if isinstance(role_value, dict)
                else str(role_value or "")
            )
            usage_class = (
                USAGE_MUSIC_SOUNDTRACK
                if role == "soundtrack" or media_type == 3
                else USAGE_SOCIAL_POST
            )
            if usage_class == USAGE_MUSIC_SOUNDTRACK and not audio_edit_recipe:
                raise ValueError("soundtrack asset requires canonical audio_edit_recipe evidence")
            if not usage_permitted(rights, usage_class):
                raise ValueError(f"asset {asset_id} does not permit usage class {usage_class}")

            rights_version = int(row.get("rights_version") or 1)
            economics_version = int(row.get("economics_version") or 1)
            originality = int(row.get("originality_status") or ORIGINALITY_ORIGINAL)
            if originality == ORIGINALITY_DERIVATIVE:
                contains_derivatives = True

            share_bps = 0
            beneficiary = None
            # Post display/cache URLs describe the top-level visual output, not the
            # protected catalog M4A. Soundtrack identity is already resolved and is
            # bound by the canonical AudioEditRecipe evidence instead.
            media_url = (
                None
                if usage_class == USAGE_MUSIC_SOUNDTRACK
                else (media_urls[idx] if idx < len(media_urls) else None)
            )
            if media_url:
                analysis = await self.analysis.analyze(
                    self.network,
                    post_id,
                    media_url,
                    idx,
                    int(row.get("media_type") or 1),
                    creator_wallet_address=post.get("creator_address"),
                )
                submission = self.decisions.build_submission(post, analysis)
                reasoning_parts.append(submission.reasoning)
                if submission.original_creator:
                    beneficiary = submission.original_creator
                    share_bps = max(share_bps, submission.highest_similarity_score)
                    derivative_target = submission.derivative_redirection_target
                if submission.needs_review:
                    contains_unresolved = True

            asset_inputs.append(
                AssetVersionInput(
                    asset_id=asset_id,
                    rights_version=rights_version,
                    economics_version=economics_version,
                    usage_class=usage_class,
                    share_bps=share_bps,
                    beneficiary=beneficiary,
                    source_asset_id=asset_id,
                )
            )
            if share_bps > 0 and beneficiary:
                manifest_entries.append(asset_inputs[-1])

            self.asset_usages.record(
                self.network,
                {
                    "asset_id": asset_id,
                    "container_id": post_id,
                    "container_type": 1,
                    "usage_class": usage_class,
                    "position": idx,
                },
            )

        cfg = self.submitter.get_poc_config()
        self.config_cache.set(self.network, cfg)
        reasoning = truncate_reasoning("; ".join(reasoning_parts) or "Composition analysis complete.", int(cfg.get("max_reasoning_length") or 5000))
        evidence = truncate_evidence_urls(evidence, int(cfg.get("max_evidence_urls") or 10)) or None

        spt_pool_id = payload.get("spt_id")
        if not spt_pool_id and self.submitter.client:
            fields, _ = self.submitter.client.fetch_post_fields(post_id)
            spt_pool_id = self.submitter.client.extract_spt_pool_id(fields or {})

        composition = build_composition_submission(
            post_id=post_id,
            assets=asset_inputs,
            manifest_entries=manifest_entries or None,
            derivative_redirection_target=derivative_target,
            max_embedded_asset_redirect_bps=int(cfg.get("max_embedded_asset_redirect_bps") or 5000),
            contains_derivatives=contains_derivatives,
            contains_unresolved_assets=contains_unresolved,
            reasoning=reasoning,
            evidence_urls=evidence,
            spt_pool_id=str(spt_pool_id) if spt_pool_id else None,
        )
        move_call = build_analyze_post_composition_move_call(composition)
        tx = self.submitter.submit_analyze_post_composition(move_call)
        tx_digest = tx.get("tx_hash")

        bindings: list[EmbeddedAssetBinding] = []
        oracle_address = os.getenv("MYSO_POC_ORACLE_ADDRESS", "")
        if not oracle_address and self.submitter.client:
            oracle_address = self.submitter.client.wallet.get_address()
        for idx, asset_id in enumerate(media_asset_ids):
            media_type = int(
                (self.media_assets.get(self.network, str(asset_id)) or {}).get("media_type") or 1
            )
            usage_class = USAGE_MUSIC_SOUNDTRACK if media_type == 3 else USAGE_SOCIAL_POST
            component = (
                MEDIA_COMPONENT_VIDEO
                if media_type == 2
                else MEDIA_COMPONENT_AUDIO
                if media_type == 3
                else MEDIA_COMPONENT_IMAGE
            )
            bindings.append(
                EmbeddedAssetBinding(
                    binding_id=deterministic_binding_id(post_id, str(asset_id), usage_class),
                    source_asset_id=str(asset_id),
                    usage_class=usage_class,
                    stem=0,
                    media_component=component,
                    evidence_commitment=(
                        audio_edit_recipe_commitment(audio_edit_recipe)
                        if usage_class == USAGE_MUSIC_SOUNDTRACK
                        else None
                    ),
                )
            )
        if bindings and oracle_address:
            bind_mc = build_record_embedded_bindings_move_call(
                oracle_address=oracle_address,
                post_id=post_id,
                bindings=bindings,
            )
            bind_tx = self.submitter.submit_move_call(bind_mc)
            if composition.manifest:
                try:
                    manifest_mc = build_submit_candidate_revenue_manifest_move_call(
                        oracle_address=oracle_address,
                        post_id=post_id,
                        manifest_entries=composition.manifest.get("entries") or [],
                        manifest_version=1,
                    )
                    self.submitter.submit_move_call(manifest_mc)
                except Exception as exc:
                    logger.warning("candidate manifest submit failed: %s", exc)
            self.jobs.enqueue(
                self.network,
                post_id,
                job_type="refresh_post_usage_decisions",
                payload={
                    "post_id": post_id,
                    "bindings": [
                        {
                            "binding_id": b.binding_id,
                            "source_asset_id": b.source_asset_id,
                            "usage_class": b.usage_class,
                            "stem": b.stem,
                            "media_component": b.media_component,
                            "evidence_commitment": (
                                b.evidence_commitment.hex() if b.evidence_commitment else None
                            ),
                        }
                        for b in bindings
                    ],
                    "composition_tx": tx_digest,
                    "bindings_tx": bind_tx.get("tx_hash"),
                },
            )

        self.composition_records.insert(
            self.network,
            {
                "post_id": post_id,
                "composition_status": composition.composition_status,
                "monetization_status": composition.monetization_status,
                "analysis_json": composition.analysis,
                "manifest_json": composition.manifest,
                "reasoning": reasoning,
                "evidence_urls": evidence or [],
                "contains_derivatives": contains_derivatives,
                "contains_unresolved_assets": contains_unresolved,
                "tx_digest": tx_digest,
            },
        )
        self.posts.update_status(
            self.network,
            post_id,
            analysis_status="attested",
            tx_digest=tx_digest,
        )
        await event_bus.publish(
            "post.attestation.submitted",
            {"network": self.network, "post_id": post_id, "tx_digest": tx_digest, "mode": "composition"},
        )
        await event_bus.publish(
            "post.attestation.confirmed",
            {
                "network": self.network,
                "post_id": post_id,
                "tx_digest": tx_digest,
                "composition_status": composition.composition_status,
                "monetization_status": composition.monetization_status,
            },
        )

    async def _process_claim(self, job: dict) -> dict:
        payload = job.get("payload") or {}
        if isinstance(payload, str):
            import json

            payload = json.loads(payload)
        identity_hash = payload.get("identity_hash")
        claimant = payload.get("claimant_address")
        beneficiary_id = payload.get("beneficiary_id")
        if not identity_hash or not claimant:
            raise ValueError("claim_beneficiary job requires identity_hash and claimant_address")

        if not beneficiary_id:
            from app.services.poc_chain_helpers import resolve_beneficiary_object_for_identity

            beneficiary_id = resolve_beneficiary_object_for_identity(
                self.profile, str(identity_hash)
            )
        if not beneficiary_id:
            raise ValueError(f"Could not resolve beneficiary object for identity {identity_hash}")

        from app.services.poc_identity import IdentityVerifier

        verifier = IdentityVerifier(self.network)
        verified = verifier.verify_claim(
            beneficiary_id=str(beneficiary_id),
            wallet=str(claimant),
            identity_hash=str(identity_hash),
            attested_x_handle=payload.get("attested_x_handle"),
            oauth_token=payload.get("oauth_token"),
            session_jwt=payload.get("session_jwt"),
            mock_headers=payload.get("mock_headers") or {},
            display_name=str(payload.get("display_name") or ""),
            bio=str(payload.get("bio") or ""),
            profile_picture_url=str(payload.get("profile_picture_url") or ""),
            cover_photo_url=str(payload.get("cover_photo_url") or ""),
        )

        result = await self.beneficiaries.claim(
            str(identity_hash),
            str(claimant),
            beneficiary_id=str(beneficiary_id),
            evidence_hash=verified.evidence_hash,
            attested_x_handle=verified.attested_x_handle,
            display_name=verified.display_name,
            bio=verified.bio,
            profile_picture_url=verified.profile_picture_url,
            cover_photo_url=verified.cover_photo_url,
        )
        tx = result.get("tx_hash")
        discovery_asset_id = payload.get("discovery_asset_id")
        await self.vault_lifecycle.mark_claimed(
            identity_hash=str(identity_hash),
            claimant_address=str(claimant),
            discovery_asset_id=str(discovery_asset_id) if discovery_asset_id else None,
        )
        await event_bus.publish(
            "beneficiary.claim.completed",
            {
                "network": self.network,
                "identity_hash": identity_hash,
                "claimant_address": claimant,
                "tx_digest": tx,
                "verifier": verified.verifier,
                **(result.get("claim_outcome") or {}),
            },
        )
        return {"tx_digest": tx, **(result.get("claim_outcome") or {})}

    async def _process_beneficiary_transfer_proposal(self, job: dict) -> dict:
        payload = job.get("payload") or {}
        if isinstance(payload, str):
            import json

            payload = json.loads(payload)
        identity_hash = str(payload.get("identity_hash") or "")
        beneficiary_id = str(payload.get("beneficiary_id") or "")
        verified_wallet = str(payload.get("verified_wallet") or "")
        recipient = str(payload.get("recipient_address") or "")
        attested_handle = str(payload.get("attested_x_handle") or "")
        evidence_hash_hex = str(payload.get("evidence_hash") or "")
        if not all((identity_hash, beneficiary_id, verified_wallet, recipient, attested_handle, evidence_hash_hex)):
            raise ValueError("beneficiary transfer proposal requires identity, beneficiary, verified wallet and recipient")
        from app.services.poc_identity import IdentityVerifier

        verified = IdentityVerifier(self.network).verify_beneficiary_transfer(
            beneficiary_id=beneficiary_id,
            verified_wallet=verified_wallet,
            recipient=recipient,
            identity_hash=identity_hash,
            attested_x_handle=attested_handle,
            evidence_hash=evidence_hash_hex,
        )
        result = await self.beneficiaries.propose_transfer(
            beneficiary_id=beneficiary_id,
            recipient=recipient,
            evidence_hash=verified.evidence_hash,
            attested_x_handle=verified.attested_x_handle,
            identity_hash=identity_hash,
        )
        await event_bus.publish(
            "beneficiary.transfer.proposal.completed",
            {
                "network": self.network,
                "identity_hash": identity_hash,
                "beneficiary_id": beneficiary_id,
                "verified_wallet": verified_wallet,
                "recipient": recipient,
                "tx_digest": result.get("tx_hash"),
                "verifier": verified.verifier,
            },
        )
        return {
            "tx_digest": result.get("tx_hash"),
            "beneficiary_id": beneficiary_id,
            "recipient_address": recipient,
        }


async def run_oracle_workers(networks: list[str]) -> None:
    workers = [OracleWorker(n) for n in networks]
    await asyncio.gather(*(w.run_forever() for w in workers))
