"""Discovery scheduler — polls creative media sources and enqueues embed jobs."""

from __future__ import annotations

import os
import uuid

import structlog

from app.discovery.store import DiscoveryStore
from app.discovery.sources.config_loader import SourceConfig, default_sources_config_path, load_sources_config
from app.discovery.sources.registry import discover_source

logger = structlog.get_logger()

CORPUS_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def discovery_enabled() -> bool:
    return os.getenv("DISCOVERY_ENABLED", "true").lower() in ("1", "true", "yes")


def embed_enabled() -> bool:
    return os.getenv("DISCOVERY_EMBED_ENABLED", "true").lower() in ("1", "true", "yes")


def poll_interval_seconds() -> int:
    return int(os.getenv("DISCOVERY_POLL_INTERVAL_SECONDS", "30"))


def _source_uuid(source_id: str) -> str:
    return str(uuid.uuid5(CORPUS_NAMESPACE, source_id))


class DiscoveryScheduler:
    def __init__(self) -> None:
        self.repo = DiscoveryStore()

    def poll_once(self, config_path=None) -> dict[str, int]:
        stats = {"sources": 0, "records": 0, "assets_new": 0, "jobs_enqueued": 0}
        if not discovery_enabled():
            return stats
        sources = load_sources_config(config_path or default_sources_config_path())
        for source in sources:
            if not source.enabled:
                continue
            stats["sources"] += 1
            source_pk = self.repo.upsert_source(
                source_id=_source_uuid(source.id),
                adapter_type=source.adapter_type,
                domain=source.domain,
                trust_score=source.trust_score,
                enabled=source.enabled,
                config=source.raw,
            )
            self.repo.touch_source_polled(source_pk)
            records = discover_source(source)
            stats["records"] += len(records)
            for record in records:
                asset_id, created = self._ingest_record(source, source_pk, record)
                if created:
                    stats["assets_new"] += 1
                if embed_enabled() and self._should_embed(source, record):
                    job_id = self.repo.enqueue_embed_job(asset_id, priority_score=int(source.trust_score * 1000))
                    if job_id:
                        stats["jobs_enqueued"] += 1
        return stats

    def _ingest_record(self, source: SourceConfig, source_pk: str, record: dict) -> tuple[str, bool]:
        return self.repo.upsert_asset(
            source_id=source_pk,
            external_source_url=record["external_source_url"],
            media_type=record["media_type"],
            content_kind=record.get("content_kind") or "media",
            trust_score=float(record.get("trust_score") or source.trust_score),
            creator_x_handle=record.get("creator_x_handle"),
            creator_confidence=float(record.get("creator_confidence") or 0),
            metadata=record.get("metadata") or {},
            priority_score=int(float(record.get("trust_score") or source.trust_score) * 1000),
        )

    @staticmethod
    def _should_embed(source: SourceConfig, record: dict) -> bool:
        return (
            source.domain == "creative"
            and (record.get("content_kind") or source.content_kind) == "media"
        )
