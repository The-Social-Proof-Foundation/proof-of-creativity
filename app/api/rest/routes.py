"""REST routes for oracle status and proof retrieval."""

from __future__ import annotations

import asyncio
import os
import json
import time

from fastapi import APIRouter, HTTPException, Query, Request
from psycopg2 import extras
from pydantic import BaseModel

from app.core.database import get_db_connection

from app.chain.networks import SUPPORTED_NETWORKS, active_networks, load_network_profile
from app.db.oracle_repository import (
    AttestationRepository,
    ChainPostRepository,
    CheckpointRepository,
    ConfigCacheRepository,
    JobRepository,
    UsernameBeneficiaryRepository,
)
from app.chain.grpc_client import (
    resolve_checkpoint_stream_id,
    resolve_event_stream_id,
    resolve_sync_mode,
)
from app.chain.grpc_v2_client import GrpcV2Client
from app.network_config import get_settings, load_network_profile
from app.services.myso_client import init_myso_client
from app.discovery.identity import identity_hash_from_x_handle
from app.services.identity_verification_client import IdentityVerificationClient
from app.services.poc_identity import resolve_identity_verifier_mode

router = APIRouter(prefix="/oracle", tags=["oracle"])

posts_repo = ChainPostRepository()
checkpoints_repo = CheckpointRepository()
jobs_repo = JobRepository()
attestations_repo = AttestationRepository()
config_repo = ConfigCacheRepository()
beneficiaries_repo = UsernameBeneficiaryRepository()


class BeneficiaryClaimRequest(BaseModel):
    claimant_address: str
    beneficiary_id: str | None = None
    attested_x_handle: str | None = None
    display_name: str = ""
    bio: str = ""
    profile_picture_url: str = ""
    cover_photo_url: str = ""
    oauth_token: str | None = None
    session_jwt: str | None = None


class PocClaimOAuthStartRequest(BaseModel):
    claimant_address: str
    beneficiary_id: str


def _network_param(network: str | None) -> str:
    if network:
        return network
    return active_networks()[0]


@router.get("/networks")
async def list_networks():
    settings = get_settings()
    return {
        "supported": list(SUPPORTED_NETWORKS),
        "active": active_networks(),
        "default": settings.myso_network,
    }


@router.get("/sync/status")
async def sync_status(network: str | None = Query(default=None)):
    net = _network_param(network)
    profile = load_network_profile(net)
    stream_id = resolve_checkpoint_stream_id(profile)
    package_id = resolve_event_stream_id(profile)
    sync_mode = resolve_sync_mode(profile)
    cp = checkpoints_repo.get(net, stream_id) or {}
    checkpoint = int(cp.get("checkpoint_sequence") or 0)
    chain_tip = 0
    lag = 0

    if sync_mode != "mock":
        try:
            client = GrpcV2Client(profile)
            try:
                chain_tip = client.get_chain_tip()
                lag = max(0, chain_tip - checkpoint) if chain_tip else 0
            finally:
                client.close()
        except Exception:
            pass

    return {
        "network": net,
        "stream_id": stream_id,
        "package_id": package_id,
        "sync_mode": sync_mode,
        "checkpoint": checkpoint,
        "chain_tip": chain_tip,
        "last_transaction_digest": cp.get("last_transaction_digest"),
        "pending_jobs": jobs_repo.pending_count(net),
        "lag_checkpoints": lag,
        "lag": lag,
        "highest_indexed_checkpoint": chain_tip,
        "last_event_at": cp.get("updated_at"),
        "mock_mode": profile.grpc_sync.mock_mode,
    }


@router.get("/posts/{post_id}")
async def get_post(post_id: str, network: str | None = Query(default=None)):
    net = _network_param(network)
    post = posts_repo.get(net, post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    return post


@router.get("/posts/{post_id}/proof")
async def get_proof(post_id: str, network: str | None = Query(default=None)):
    net = _network_param(network)
    post = posts_repo.get(net, post_id)
    if not post or not post.get("proof_bundle_uri"):
        raise HTTPException(status_code=404, detail="Proof bundle not found")
    uri = post["proof_bundle_uri"]
    if uri.startswith("file://"):
        from pathlib import Path

        path = Path(uri.replace("file://", ""))
        if not path.exists():
            raise HTTPException(status_code=404, detail="Proof file missing")
        return json.loads(path.read_text(encoding="utf-8"))
    return {"proof_bundle_uri": uri}


@router.get("/posts/{post_id}/matches")
async def get_matches(post_id: str, network: str | None = Query(default=None)):
    net = _network_param(network)
    attestations = attestations_repo.list_for_post(net, post_id)
    return {"network": net, "post_id": post_id, "attestations": attestations}


@router.get("/beneficiaries/{identity_hash}")
async def get_beneficiary(identity_hash: str, network: str | None = Query(default=None)):
    net = _network_param(network)
    record = beneficiaries_repo.get(net, identity_hash)
    if not record:
        raise HTTPException(status_code=404, detail="Beneficiary not found")
    return record


@router.get("/config")
async def get_config(network: str | None = Query(default=None)):
    net = _network_param(network)
    cached = config_repo.get(net)
    if cached:
        return {"network": net, "config": cached, "source": "cache"}
    profile = load_network_profile(net)
    from app.chain.rpc_client import TransactionSubmitter

    submitter = TransactionSubmitter(net)
    cfg = submitter.get_poc_config()
    config_repo.set(net, cfg)
    return {"network": net, "config": cfg, "source": "rpc"}


@router.get("/chain/beneficiary-vault")
async def get_beneficiary_vault(
    beneficiary: str = Query(...),
    network: str | None = Query(default=None),
):
    net = _network_param(network)
    profile = load_network_profile(net)
    from app.services.poc_chain_helpers import resolve_beneficiary_vault_id

    vault_id = resolve_beneficiary_vault_id(profile, beneficiary)
    return {"network": net, "beneficiary": beneficiary, "vault_id": vault_id}


@router.get("/chain/username-beneficiary/{beneficiary_id}")
async def get_username_beneficiary_on_chain(
    beneficiary_id: str,
    network: str | None = Query(default=None),
):
    net = _network_param(network)
    profile = load_network_profile(net)
    from app.services.poc_chain_helpers import fetch_username_beneficiary_fields

    fields = fetch_username_beneficiary_fields(profile, beneficiary_id)
    return {"network": net, "beneficiary_id": beneficiary_id, "fields": fields}


async def _wait_for_job(job_id: str, timeout_sec: float = 120.0) -> dict:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        job = jobs_repo.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        status = job.get("status")
        if status in ("completed", "failed"):
            return job
        await asyncio.sleep(0.5)
    raise HTTPException(status_code=504, detail=f"Job {job_id} did not complete within {timeout_sec}s")


@router.get("/beneficiaries/{identity_hash}/claim/status")
async def get_claim_status(
    identity_hash: str,
    request: Request,
    claimant_address: str = Query(...),
    network: str | None = Query(default=None),
):
    net = _network_param(network)
    wallet = claimant_address.strip()
    if not wallet:
        raise HTTPException(status_code=400, detail="claimant_address is required")

    record = beneficiaries_repo.get(net, identity_hash)
    claimable = bool(record and record.get("provision_tx_digest") and not record.get("claim_tx_digest"))
    required_x_handle = (record or {}).get("username")

    session_jwt = request.headers.get("Authorization", "").removeprefix("Bearer ").strip() or None
    mode = resolve_identity_verifier_mode(net)
    oauth_required = mode in ("myso-identity", "myso_identity", "identity-verification")
    oauth_complete = False
    authorize_url = None
    attested_x_handle = None

    if oauth_required:
        client = IdentityVerificationClient()
        if client.enabled:
            try:
                status = client.get_claim_status_sync(
                    identity_hash=identity_hash,
                    wallet=wallet,
                    session_jwt=session_jwt,
                )
                oauth_required = status.oauth_required
                oauth_complete = status.oauth_complete
                authorize_url = status.authorize_url
                attested_x_handle = status.attested_x_handle
            except Exception as exc:
                oauth_required = True
                authorize_url = None
                attested_x_handle = None
                oauth_complete = False
                _ = exc

    return {
        "network": net,
        "identity_hash": identity_hash,
        "claimable": claimable,
        "oauth_required": oauth_required,
        "oauth_complete": oauth_complete,
        "required_x_handle": required_x_handle,
        "authorize_url": authorize_url,
        "attested_x_handle": attested_x_handle,
        "beneficiary": record,
    }


@router.post("/beneficiaries/{identity_hash}/claim/oauth/start")
async def start_claim_oauth(
    identity_hash: str,
    request: Request,
    body: PocClaimOAuthStartRequest,
    network: str | None = Query(default=None),
):
    net = _network_param(network)
    session_jwt = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not session_jwt:
        raise HTTPException(status_code=401, detail="Authorization Bearer session JWT required")

    client = IdentityVerificationClient()
    if not client.enabled:
        raise HTTPException(status_code=503, detail="identity verification service not configured")

    authorize_url = await client.connect_for_poc_claim(
        identity_hash=identity_hash,
        beneficiary_id=body.beneficiary_id,
        wallet=body.claimant_address,
        session_jwt=session_jwt,
    )
    return {
        "network": net,
        "identity_hash": identity_hash,
        "authorize_url": authorize_url,
    }


@router.post("/beneficiaries/{identity_hash}/claim")
async def enqueue_claim(
    identity_hash: str,
    request: Request,
    body: BeneficiaryClaimRequest | None = None,
    claimant_address: str | None = Query(default=None),
    sync: bool = Query(default=False, description="Wait for job completion (e2e)"),
    network: str | None = Query(default=None),
):
    net = _network_param(network)
    claim_body = body or BeneficiaryClaimRequest(claimant_address=claimant_address or "")
    wallet = (claim_body.claimant_address or claimant_address or "").strip()
    if not wallet:
        raise HTTPException(status_code=400, detail="claimant_address is required")

    mode = resolve_identity_verifier_mode(net)
    if (
        net in ("testnet", "mainnet")
        and claim_body.oauth_token
        and os.getenv("ALLOW_LEGACY_OAUTH_TOKEN", "").strip().lower() not in ("1", "true", "yes")
    ):
        raise HTTPException(
            status_code=400,
            detail="raw oauth_token is disabled; use session JWT with myso-identity-verification",
        )

    mock_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower().startswith("x-poc-mock-")
    }
    session_jwt = (
        claim_body.session_jwt
        or request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        or None
    )

    payload = {
        "identity_hash": identity_hash,
        "claimant_address": wallet,
        "beneficiary_id": claim_body.beneficiary_id,
        "attested_x_handle": claim_body.attested_x_handle,
        "display_name": claim_body.display_name,
        "bio": claim_body.bio,
        "profile_picture_url": claim_body.profile_picture_url,
        "cover_photo_url": claim_body.cover_photo_url,
        "oauth_token": claim_body.oauth_token if mode in ("x-oauth", "x_oauth", "oauth") else None,
        "session_jwt": session_jwt,
        "mock_headers": mock_headers,
    }
    job_id = jobs_repo.enqueue(
        net,
        post_id=f"claim:{identity_hash}",
        job_type="claim_beneficiary",
        payload=payload,
    )
    if not sync:
        return {"network": net, "job_id": job_id, "status": "queued"}

    job = await _wait_for_job(job_id)
    if job.get("status") == "failed":
        raise HTTPException(status_code=500, detail=job.get("last_error") or "claim job failed")
    record = beneficiaries_repo.get(net, identity_hash) or {}
    return {
        "network": net,
        "job_id": job_id,
        "status": "completed",
        "claim_tx_digest": record.get("claim_tx_digest"),
        "beneficiary": record,
    }


class ReviewResolveRequest(BaseModel):
    action: str
    identity_hash: str | None = None
    creator_x_handle: str | None = None
    notes: str | None = None
    resolved_by: str | None = None


@router.get("/reviews")
async def list_reviews(
    network: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    net = _network_param(network)
    items = posts_repo.list_needs_review(net, limit=limit, offset=offset)
    return {"network": net, "items": items, "limit": limit, "offset": offset}


@router.get("/reviews/{post_id}")
async def get_review(post_id: str, network: str | None = Query(default=None)):
    net = _network_param(network)
    post = posts_repo.get(net, post_id)
    if not post or post.get("analysis_status") != "needs_review":
        raise HTTPException(status_code=404, detail="Review item not found")
    from app.db.discovery_repository import ProvenanceHitRepository

    hits = []
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT * FROM provenance_hits
                WHERE network = %s AND post_id = %s
                ORDER BY created_at DESC LIMIT 10
                """,
                (net, post_id),
            )
            hits = [dict(r) for r in cur.fetchall()]
    return {"network": net, "post": post, "provenance_hits": hits}


@router.post("/reviews/{post_id}/resolve")
async def resolve_review(
    post_id: str,
    body: ReviewResolveRequest,
    network: str | None = Query(default=None),
):
    net = _network_param(network)
    post = posts_repo.get(net, post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    action = body.action.strip().lower()
    if action not in ("approve_escrow", "reject", "requeue"):
        raise HTTPException(status_code=400, detail="Invalid action")

    posts_repo.resolve_review(
        net,
        post_id,
        action=action,
        resolved_by=body.resolved_by,
        notes=body.notes,
    )

    if action == "approve_escrow":
        media_urls = post.get("media_urls") or []
        if isinstance(media_urls, str):
            media_urls = json.loads(media_urls)
        media_types = post.get("media_types") or []
        if isinstance(media_types, str):
            media_types = json.loads(media_types)
        media_url = media_urls[0] if media_urls else ""
        media_type = int(media_types[0]) if media_types else 1
        override = {}
        if body.identity_hash:
            override["identity_hash"] = body.identity_hash
        if body.creator_x_handle:
            override["creator_x_handle"] = body.creator_x_handle
            override.setdefault("identity_hash", identity_hash_from_x_handle(body.creator_x_handle))
        jobs_repo.enqueue(
            net,
            post_id=post_id,
            job_type="analyze_post",
            media_url=media_url,
            media_type=media_type,
            payload={"review_override": override} if override else {},
        )
    elif action == "requeue":
        media_urls = post.get("media_urls") or []
        if isinstance(media_urls, str):
            media_urls = json.loads(media_urls)
        media_types = post.get("media_types") or []
        if isinstance(media_types, str):
            media_types = json.loads(media_types)
        jobs_repo.enqueue(
            net,
            post_id=post_id,
            job_type="analyze_post",
            media_url=media_urls[0] if media_urls else "",
            media_type=int(media_types[0]) if media_types else 1,
        )

    return {"network": net, "post_id": post_id, "action": action, "status": "accepted"}


@router.get("/registry/username/{handle}")
async def registry_username_lookup(handle: str, network: str | None = Query(default=None)):
    net = _network_param(network)
    ih = identity_hash_from_x_handle(handle)
    record = beneficiaries_repo.get(net, ih)
    profile = load_network_profile(net)
    from app.services.poc_chain_helpers import (
        fetch_username_beneficiary_fields,
        resolve_beneficiary_object_for_identity,
    )

    beneficiary_id = resolve_beneficiary_object_for_identity(profile, ih)
    fields = fetch_username_beneficiary_fields(profile, beneficiary_id) if beneficiary_id else {}
    return {
        "network": net,
        "handle": handle,
        "identity_hash": ih,
        "beneficiary": record,
        "beneficiary_id": beneficiary_id,
        "on_chain_fields": fields,
    }


@router.get("/health")
async def oracle_health(network: str | None = Query(default=None)):
    net = _network_param(network)
    profile = load_network_profile(net)
    stream_id = resolve_checkpoint_stream_id(profile)
    package_id = resolve_event_stream_id(profile)
    sync_mode = resolve_sync_mode(profile)
    client = init_myso_client()
    return {
        "network": net,
        "stream_id": stream_id,
        "package_id": package_id,
        "sync_mode": sync_mode,
        "sync": checkpoints_repo.get(net, stream_id) or {},
        "pending_jobs": jobs_repo.pending_count(net),
        "mysocial": mysocial_readiness_payload(client),
        "websocket_connections": 0,
        "mock_mode": profile.grpc_sync.mock_mode,
    }
