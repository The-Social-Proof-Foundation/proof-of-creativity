"""Embedding context for platform vs discovered corpus rows."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def default_embedding_model() -> str:
    return os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")


def default_embedding_version() -> str:
    explicit = os.getenv("DISCOVERY_ACTIVE_EMBEDDING_VERSION", "").strip()
    if explicit:
        return explicit
    model = default_embedding_model().replace("/", "-")
    return f"{model}-v1"


def default_embedding_dimension() -> int:
    return int(os.getenv("EMBEDDING_DIMENSION", "512"))


@dataclass
class EmbeddingContext:
    corpus_scope: str = "platform"
    discovery_asset_id: str | None = None
    embedding_model: str = field(default_factory=default_embedding_model)
    embedding_version: str = field(default_factory=default_embedding_version)
    embedding_dimension: int = field(default_factory=default_embedding_dimension)
    creator_x_handle: str | None = None
    creator_confidence: float = 0.0
    identity_hash: str | None = None
    work_confidence: float = 0.0

    def provenance_metadata(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "corpus_scope": self.corpus_scope,
            "embedding_model": self.embedding_model,
            "embedding_version": self.embedding_version,
            "embedding_dimension": self.embedding_dimension,
            "visibility": "internal" if self.corpus_scope == "discovered" else "platform",
        }
        if self.discovery_asset_id:
            meta["discovery_asset_id"] = self.discovery_asset_id
        if self.identity_hash:
            meta["identity_hash"] = self.identity_hash
        if self.creator_x_handle:
            meta["creator_x_handle"] = self.creator_x_handle
        if self.creator_confidence:
            meta["creator_confidence"] = self.creator_confidence
        if self.work_confidence:
            meta["work_confidence"] = self.work_confidence
        return meta

    def corpus_row_kwargs(self) -> dict[str, Any]:
        return {
            "corpus_scope": self.corpus_scope,
            "discovery_asset_id": self.discovery_asset_id,
            "embedding_model": self.embedding_model,
            "embedding_version": self.embedding_version,
            "embedding_dimension": self.embedding_dimension,
            "embedding_created_at": datetime.now(timezone.utc),
        }


def discovered_context(
    *,
    discovery_asset_id: str,
    embedding_version: str | None = None,
    creator_x_handle: str | None = None,
    creator_confidence: float = 0.0,
    identity_hash: str | None = None,
) -> EmbeddingContext:
    return EmbeddingContext(
        corpus_scope="discovered",
        discovery_asset_id=discovery_asset_id,
        embedding_version=embedding_version or default_embedding_version(),
        creator_x_handle=creator_x_handle,
        creator_confidence=creator_confidence,
        identity_hash=identity_hash,
    )
