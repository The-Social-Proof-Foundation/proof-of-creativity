"""Unit tests for post-first media resolution (metadata commitments + event helpers)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.chain.event_parser import (
    infer_primary_media_type,
    parse_post_metadata_commitments,
    parse_post_created,
    parse_post_media_clips,
    plan_post_media_jobs,
)
from app.chain.ptb_builder import extract_event_field, extract_media_asset_resolved_id


def test_parse_post_metadata_commitments_from_snake_case():
    meta = json.dumps(
        {
            "asset_id": "vid-123",
            "content_commitment": "0xabc",
            "observed_fingerprint_commitment": "0xdef",
        }
    )
    parsed = parse_post_metadata_commitments(meta)
    assert parsed is not None
    assert parsed["content_commitment"] == "abc"
    assert parsed["observed_fingerprint_commitment"] == "def"
    assert parsed["asset_id"] == "vid-123"


def test_parse_post_metadata_commitments_missing_fields():
    assert parse_post_metadata_commitments(json.dumps({"asset_id": "x"})) is None
    assert parse_post_metadata_commitments(None) is None


def test_parse_post_created_includes_metadata_json():
    parsed = parse_post_created(
        {
            "post_id": "0xpost",
            "owner": "0xowner",
            "media_urls": ["https://cdn/x/master.m3u8"],
            "metadata_json": '{"content_commitment":"aa","observed_fingerprint_commitment":"bb"}',
        }
    )
    assert parsed is not None
    assert parsed["metadata_json"] is not None


def test_plan_resolve_even_when_soundtrack_ids_exist():
    meta = json.dumps(
        {
            "content_commitment": "aa",
            "observed_fingerprint_commitment": "bb",
        }
    )
    plan = plan_post_media_jobs(
        media_urls=["https://cdn/v/master.m3u8"],
        media_types=[2],
        media_asset_ids=["0xsoundtrack"],
        metadata_json=meta,
    )
    assert plan["action"] == "resolve_post_media"
    assert plan["existing_media_asset_ids"] == ["0xsoundtrack"]
    assert plan["clips"][0]["content_commitment"] == "aa"


def test_plan_compose_when_all_clips_already_have_ids():
    meta = json.dumps(
        {
            "media_commitments": [
                {"media_asset_id": "0xvideo", "url": "https://cdn/v.mp4"},
            ]
        }
    )
    plan = plan_post_media_jobs(
        media_urls=["https://cdn/v.mp4"],
        media_types=[2],
        media_asset_ids=["0xvideo"],
        metadata_json=meta,
    )
    assert plan["action"] == "analyze_composition"
    assert plan["media_asset_ids"] == ["0xvideo"]


def test_parse_media_commitments_array():
    meta = json.dumps(
        {
            "media_commitments": [
                {
                    "url": "https://cdn/a.mp4",
                    "content_commitment": "0x11",
                    "observed_fingerprint_commitment": "0x22",
                    "media_type": 2,
                },
                {
                    "url": "https://cdn/b.jpg",
                    "content_commitment": "0x33",
                    "observed_fingerprint_commitment": "0x44",
                    "media_type": 1,
                },
            ]
        }
    )
    clips = parse_post_media_clips(["https://cdn/a.mp4", "https://cdn/b.jpg"], [2, 1], meta)
    assert len(clips) == 2
    assert clips[1]["content_commitment"] == "33"


def test_infer_primary_media_type_prefers_video():
    urls = ["https://cdn/preview.jpg", "https://cdn/v/master.m3u8"]
    assert infer_primary_media_type(urls, []) == 2


def test_extract_media_asset_resolved_id_from_event():
    tx = {
        "events": [
            {
                "type": "0xpkg::media_asset::MediaAssetResolvedEvent",
                "parsedJson": {
                    "fields": {
                        "media_asset_id": "0xasset",
                    }
                },
            }
        ]
    }
    assert extract_media_asset_resolved_id(tx) == "0xasset"
    assert extract_event_field(tx, "MediaAssetResolvedEvent", "media_asset_id") == "0xasset"


@pytest.mark.asyncio
async def test_resolve_post_media_enqueues_composition():
    from app.workers import oracle_worker as ow

    worker = ow.OracleWorker.__new__(ow.OracleWorker)
    worker.network = "localnet"
    worker.submitter = MagicMock()
    worker.jobs = MagicMock()
    worker.pending_assets = MagicMock()
    worker.pending_assets.find_by_request.return_value = None
    worker.fingerprint_obs = MagicMock()
    worker.fingerprint_obs.find_asset_for_fingerprint.return_value = None
    worker.media_assets = MagicMock()
    worker.media_assets.find_by_fingerprint.return_value = None

    worker.submitter.submit_move_call.return_value = {
        "tx_hash": "0xsubmit",
        "events": [
            {
                "type": "::MediaResolutionRequestedEvent",
                "parsedJson": {"fields": {"request_id": "0xreq"}},
            }
        ],
    }
    worker.submitter.submit_finalize_media_asset.return_value = {
        "tx_hash": "0xfinalize",
        "events": [
            {
                "type": "::MediaAssetResolvedEvent",
                "parsedJson": {"fields": {"media_asset_id": "0xcanonical"}},
            }
        ],
    }

    with patch.object(ow, "_pending_first_enabled", return_value=False):
        with patch.object(ow.event_bus, "publish", new_callable=AsyncMock):
            await worker._process_resolve_post_media(
                {
                    "post_id": "0xpost",
                    "payload": {
                        "post_id": "0xpost",
                        "content_commitment": "aa" * 16,
                        "observed_fingerprint_commitment": "bb" * 16,
                        "media_type": 2,
                        "submitter": "0xowner",
                    },
                }
            )

    worker.jobs.enqueue.assert_called_once()
    call_kwargs = worker.jobs.enqueue.call_args.kwargs
    assert call_kwargs["job_type"] == "analyze_composition"
    assert call_kwargs["payload"]["media_asset_ids"] == ["0xcanonical"]


@pytest.mark.asyncio
async def test_resolve_post_media_keeps_existing_ids_and_new_clips():
    from app.workers import oracle_worker as ow

    worker = ow.OracleWorker.__new__(ow.OracleWorker)
    worker.network = "localnet"
    worker.submitter = MagicMock()
    worker.jobs = MagicMock()
    worker.pending_assets = MagicMock()
    worker.pending_assets.find_by_request.return_value = None
    worker.fingerprint_obs = MagicMock()
    worker.fingerprint_obs.find_asset_for_fingerprint.return_value = None
    worker.media_assets = MagicMock()
    worker.media_assets.find_by_fingerprint.return_value = None
    worker.submitter.submit_move_call.return_value = {
        "tx_hash": "0xsubmit",
        "events": [
            {
                "type": "::MediaResolutionRequestedEvent",
                "parsedJson": {"fields": {"request_id": "0xreq"}},
            }
        ],
    }
    worker.submitter.submit_finalize_media_asset.return_value = {
        "tx_hash": "0xfinalize",
        "events": [
            {
                "type": "::MediaAssetResolvedEvent",
                "parsedJson": {"fields": {"media_asset_id": "0xvideo"}},
            }
        ],
    }

    with patch.object(ow, "_pending_first_enabled", return_value=False):
        with patch.object(ow.event_bus, "publish", new_callable=AsyncMock):
            await worker._process_resolve_post_media(
                {
                    "post_id": "0xpost",
                    "payload": {
                        "post_id": "0xpost",
                        "existing_media_asset_ids": ["0xsoundtrack"],
                        "clips": [
                            {
                                "content_commitment": "aa" * 16,
                                "observed_fingerprint_commitment": "bb" * 16,
                                "media_type": 2,
                            }
                        ],
                        "submitter": "0xowner",
                    },
                }
            )

    ids = worker.jobs.enqueue.call_args.kwargs["payload"]["media_asset_ids"]
    assert ids == ["0xsoundtrack", "0xvideo"]
