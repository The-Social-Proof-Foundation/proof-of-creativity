"""Read-only GraphQL queries against the social indexer for oracle REST display."""

from __future__ import annotations

from typing import Any

import requests
import structlog

from app.chain.graphql_session import resolve_graphql_url
from app.network_config import NetworkProfile

logger = structlog.get_logger()


def _post_graphql(profile: NetworkProfile, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
    url = resolve_graphql_url(profile)
    resp = requests.post(
        url,
        json={"query": query, "variables": variables or {}},
        timeout=15,
    )
    resp.raise_for_status()
    body = resp.json()
    if body.get("errors"):
        logger.warning("GraphQL errors", errors=body["errors"])
    return body.get("data") or {}


def fetch_media_asset(profile: NetworkProfile, asset_id: str) -> dict[str, Any] | None:
    query = """
    query MediaAsset($id: ID!) {
      mediaAsset(id: $id) {
        mediaAssetId
        mediaType
        assetKind
        originalityStatus
        lineageParentId
        rightsVersion
        economicsVersion
        derivativeGraph {
          assetId
          parentEdges { parentAssetId childAssetId relationshipType licenseInstanceId }
          childEdges { parentAssetId childAssetId relationshipType licenseInstanceId }
        }
        resolvedPolicy {
          policyVersion
          derivativesAllowed
          commercialAllowed
          attributionRequired
        }
        detectedRelationships {
          proposalId
          accusedPendingId
          accusedAssetId
          originalAssetId
          similarityBps
          status
        }
      }
    }
    """
    data = _post_graphql(profile, query, {"id": asset_id})
    return data.get("mediaAsset")


def fetch_post_playback_policy(profile: NetworkProfile, post_id: str) -> dict[str, Any] | None:
    query = """
    query PostPlayback($id: ID!) {
      post(id: $id) {
        id
        playbackPolicy { audioMuted videoRestricted payoutRestricted }
        embeddedBindings { bindingId sourceAssetId usageClass stem mediaComponent }
        usageDecisions {
          bindingId
          playbackPermitted
          payoutPermitted
          policyPlaybackPermitted
          policyPayoutPermitted
          policyReasonCode
          policyVersionAtDecision
        }
        usageDenials { bindingId denialScope }
      }
    }
    """
    data = _post_graphql(profile, query, {"id": post_id})
    return data.get("post")


def fetch_post_enforcement_summary(profile: NetworkProfile, post_id: str) -> dict[str, Any] | None:
    return fetch_post_playback_policy(profile, post_id)


# Governance proposal lifecycle (matches on-chain governance::STATUS_*)
PROPOSAL_STATUS_COMMUNITY_VOTING = 2
PROPOSAL_STATUS_APPROVED = 3
PROPOSAL_STATUS_REJECTED = 4
PROPOSAL_STATUS_IMPLEMENTED = 5

POC_REGISTRY_TYPE = 1
POC_PROPOSAL_KIND_MEDIA_ASSET_RIGHTS = 1


def fetch_poc_governance_proposals(
    profile: NetworkProfile,
    *,
    status: int,
    limit: int = 20,
) -> list[dict[str, Any]]:
    query = """
    query PocGovernanceProposals($registryType: Int, $status: Int, $limit: Int) {
      proposals(registryType: $registryType, status: $status, limit: $limit) {
        proposalId
        referenceId
        metadataJson
        status
        votingEndTime
        title
        description
      }
    }
    """
    data = _post_graphql(
        profile,
        query,
        {
            "registryType": POC_REGISTRY_TYPE,
            "status": status,
            "limit": limit,
        },
    )
    return list(data.get("proposals") or [])


def is_media_asset_rights_proposal(proposal: dict[str, Any]) -> bool:
    metadata = proposal.get("metadataJson") or {}
    if isinstance(metadata, str):
        import json

        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    kind = metadata.get("poc_proposal_kind")
    if kind in (POC_PROPOSAL_KIND_MEDIA_ASSET_RIGHTS, "media_asset_rights", "1", 1):
        return True
    return bool(proposal.get("referenceId"))


def fetch_media_asset_usages(
    profile: NetworkProfile,
    asset_id: str,
    *,
    limit: int = 100,
) -> list[dict[str, Any]]:
    query = """
    query MediaAssetUsages($id: ID!, $limit: Int) {
      mediaAsset(id: $id) {
        mediaAssetId
        usages(limit: $limit) {
          containerId
          containerType
          position
        }
      }
    }
    """
    data = _post_graphql(profile, query, {"id": asset_id, "limit": limit})
    asset = data.get("mediaAsset") or {}
    return list(asset.get("usages") or [])
