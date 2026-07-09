"""Proof bundle generation and storage."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from app.network_config import NetworkProfile, ROOT_DIR, load_network_profile
from app.services.decision_engine import PoCSubmission
from app.services.analysis.pipeline import AnalysisResult

logger = structlog.get_logger()


class ProofBundleService:
    def __init__(self, network: str) -> None:
        self.network = network
        self.profile: NetworkProfile = load_network_profile(network)

    def build_bundle(
        self,
        post: dict[str, Any],
        analysis: AnalysisResult,
        submission: PoCSubmission,
        *,
        tx_digest: str | None = None,
    ) -> dict[str, Any]:
        expose_sources = os.getenv("DISCOVERY_EXPOSE_SOURCE_IN_EVIDENCE", "false").lower() in (
            "1",
            "true",
            "yes",
        )
        redacted_analysis = {
            "media_id": analysis.media_id,
            "highest_similarity_u64": analysis.highest_similarity_u64,
            "match_count": len(analysis.matches),
            "off_network": analysis.off_network,
            "identity_hash": analysis.identity_hash,
            "work_confidence": analysis.work_confidence,
            "creator_confidence": analysis.creator_confidence,
        }
        if expose_sources and analysis.discovery_asset_id:
            redacted_analysis["discovery_asset_id"] = analysis.discovery_asset_id
        return {
            "version": 1,
            "network": self.network,
            "post_id": analysis.post_id,
            "media_url": analysis.media_url if expose_sources else "[redacted]",
            "media_type": analysis.media_type,
            "analysis": redacted_analysis,
            "submission": {
                "highest_similarity_score": submission.highest_similarity_score,
                "original_creator": submission.original_creator,
                "derivative_redirection_target": submission.derivative_redirection_target,
                "embedded_audio_only_derivative": submission.embedded_audio_only_derivative,
                "reasoning": submission.reasoning,
            },
            "tx_digest": tx_digest,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def store(self, bundle: dict[str, Any]) -> str:
        storage = self.profile.storage
        payload = json.dumps(bundle, indent=2).encode("utf-8")
        name = f"proofs/{self.network}/{bundle['post_id']}_{uuid.uuid4().hex[:8]}.json"

        if storage.use_local or not (
            storage.use_gcs
            or storage.use_walrus
            or os.getenv("USE_CLOUDFLARE_R2", "").lower() in ("1", "true", "yes")
        ):
            base = Path(storage.local_path)
            if not base.is_absolute():
                base = ROOT_DIR / base
            base.mkdir(parents=True, exist_ok=True)
            path = base / Path(name).name
            path.write_bytes(payload)
            return f"file://{path}"

        try:
            import io

            from app.core.storage import StorageClient

            client = StorageClient()
            uri = client.upload(
                Path(name).name,
                io.BytesIO(payload),
                media_type="application/json",
            )
            if uri:
                return uri
        except Exception as exc:
            logger.warning("Remote proof storage failed; falling back to local", error=str(exc))

        base = Path(storage.local_path)
        if not base.is_absolute():
            base = ROOT_DIR / base
        base.mkdir(parents=True, exist_ok=True)
        path = base / Path(name).name
        path.write_bytes(payload)
        return f"file://{path}"
