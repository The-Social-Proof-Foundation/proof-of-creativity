"""Internal discovery embedding API."""

from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from app.discovery.embedding_service import embed_discovered_asset, verify_embed_secret
from app.discovery.bootstrap import bootstrap_status_dict

router = APIRouter(prefix="/internal/discovery", tags=["discovery-internal"])


class EmbedRequest(BaseModel):
    discovery_asset_id: str
    external_source_url: str
    media_type: str
    embedding_version: str | None = None
    creator_x_handle: str | None = None
    creator_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    creator_candidate_id: str | None = None


class EmbedResponse(BaseModel):
    media_id: str
    work_confidence: float
    embedding_version: str
    embedding_model: str
    identity_hash: str | None = None


@router.get("/status")
async def discovery_status(authorization: str | None = Header(default=None)) -> dict:
    if not verify_embed_secret(authorization):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return bootstrap_status_dict()


@router.post("/embed", response_model=EmbedResponse)
async def embed_asset(
    body: EmbedRequest,
    authorization: str | None = Header(default=None),
) -> EmbedResponse:
    if not verify_embed_secret(authorization):
        raise HTTPException(status_code=401, detail="Unauthorized")
    # validate_embed_media_type / unsupported-media 422 raised inside embed_discovered_asset
    result = await embed_discovered_asset(
        discovery_asset_id=body.discovery_asset_id,
        external_source_url=body.external_source_url,
        media_type=body.media_type,
        embedding_version=body.embedding_version,
        creator_x_handle=body.creator_x_handle,
        creator_confidence=body.creator_confidence,
        creator_candidate_id=body.creator_candidate_id,
    )
    return EmbedResponse(
        media_id=result.media_id,
        work_confidence=result.work_confidence,
        embedding_version=result.embedding_version,
        embedding_model=result.embedding_model,
        identity_hash=result.identity_hash,
    )
