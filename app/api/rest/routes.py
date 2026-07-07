"""REST routes for oracle status and proof retrieval."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Query

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
from app.services.poc_utils import mysocial_readiness_payload

router = APIRouter(prefix="/oracle", tags=["oracle"])

posts_repo = ChainPostRepository()
checkpoints_repo = CheckpointRepository()
jobs_repo = JobRepository()
attestations_repo = AttestationRepository()
config_repo = ConfigCacheRepository()
beneficiaries_repo = UsernameBeneficiaryRepository()


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


@router.post("/beneficiaries/{identity_hash}/claim")
async def enqueue_claim(
    identity_hash: str,
    claimant_address: str = Query(...),
    network: str | None = Query(default=None),
):
    net = _network_param(network)
    job_id = jobs_repo.enqueue(
        net,
        post_id=f"claim:{identity_hash}",
        job_type="claim_beneficiary",
        payload={"identity_hash": identity_hash, "claimant_address": claimant_address},
    )
    return {"network": net, "job_id": job_id, "status": "queued"}


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
