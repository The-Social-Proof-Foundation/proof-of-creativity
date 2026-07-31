"""Oracle worker — processes analyze and claim jobs."""

from __future__ import annotations

import asyncio

import structlog

from app.chain.rpc_client import TransactionSubmitter
from app.db.discovery_repository import ProvenanceHitRepository
from app.db.oracle_repository import (
    AttestationRepository,
    ChainPostRepository,
    ConfigCacheRepository,
    JobRepository,
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
from app.services.poc_utils import truncate_evidence_urls, truncate_reasoning

logger = structlog.get_logger()


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

    async def run_forever(self) -> None:
        logger.info("Oracle worker started", network=self.network)
        sem = asyncio.Semaphore(self.settings.oracle_worker_concurrency)
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
            try:
                if job_type == "claim_beneficiary":
                    await self._process_claim(job)
                else:
                    await self._process_analyze(job)
                self.jobs.complete(job_id, "completed")
            except Exception as exc:
                if job_type != "claim_beneficiary" and _is_media_not_ready(exc) and attempts < max_retries:
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

    async def _process_claim(self, job: dict) -> None:
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
            },
        )


async def run_oracle_workers(networks: list[str]) -> None:
    workers = [OracleWorker(n) for n in networks]
    await asyncio.gather(*(w.run_forever() for w in workers))
