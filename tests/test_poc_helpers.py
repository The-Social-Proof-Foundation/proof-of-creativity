"""PoC oracle utilities (pure logic, no chain)."""

import os
from unittest.mock import patch

from app.services.poc_utils import (
    OFFCHAIN_DEFAULT_POC_CONFIG,
    similarity_float_to_u64_percent,
    truncate_evidence_urls,
    truncate_reasoning,
)
from app.services.poc_submission import compose_submission_bundle


def test_mysocial_readiness_not_requested_is_ready():
    from unittest.mock import patch

    from app.services.poc_utils import mysocial_readiness_payload

    with patch.dict("os.environ", {"MYSO_INTEGRATION_ENABLED": "false"}, clear=False):
        p = mysocial_readiness_payload(None)
    assert p["integration_requested"] is False
    assert p["ready_for_submission"] is True


def test_truncation_reasoning_and_evidence():
    long = "abcdefghij"
    assert truncate_reasoning(long, max_length=10) == long
    assert truncate_reasoning(long, max_length=5) == "abcd…"
    assert truncate_evidence_urls([" a ", "", "b"], max_count=10) == ["a", "b"]
    assert truncate_evidence_urls(["x", "y", "z"], max_count=2) == ["x", "y"]


def test_check_post_preflight_mocked_rpc_shapes():
    """Guards against regressions in post.move field names used for duplicate submission."""
    from app.services.myso_client import MySocialClient

    c = MySocialClient.__new__(MySocialClient)

    c.fetch_post_fields = lambda _pid: (
        {
            "poc_outcome": "0",
            "revenue_redirect_to": None,
            "poc_badge_snapshot": None,
            "poc_badge_object_id": "0xbadge",
        },
        None,
    )
    r = MySocialClient.check_post_already_analyzed(c, "0xpost")
    assert r["already_analyzed"] is True
    forced = MySocialClient.check_post_already_analyzed(c, "0xpost", force_reanalyze=True)
    assert forced["already_analyzed"] is True

    with patch.dict(os.environ, {"MYSO_POC_ALLOW_FORCE_RESUBMIT": "true"}):
        forced_ok = MySocialClient.check_post_already_analyzed(c, "0xpost", force_reanalyze=True)
        assert forced_ok["already_analyzed"] is False

    c.fetch_post_fields = lambda _pid: (
        {
            "poc_outcome": "0",
            "revenue_redirect_to": None,
            "poc_badge_snapshot": None,
            "poc_badge_object_id": None,
            "poc_disputes_submitted": "2",
        },
        None,
    )
    cleared = MySocialClient.check_post_already_analyzed(c, "0xpost")
    assert cleared["already_analyzed"] is False
    assert cleared["poc_disputes_submitted"] == "2"
    assert cleared["supports_resubmit_when_cleared"] is True


def test_similarity_float_clamp_and_percent():
    assert similarity_float_to_u64_percent(-0.1) == 0
    assert similarity_float_to_u64_percent(float("nan")) == 0
    assert similarity_float_to_u64_percent(0.0) == 0
    assert similarity_float_to_u64_percent(0.94499) == 94
    assert similarity_float_to_u64_percent(0.999995) == 100


def test_canonical_parent_media_id_for_frame_matches():
    from app.services.poc_media_ids import (
        canonical_parent_media_id,
        corpus_parent_media_id,
        expand_media_ids_for_creator_lookup,
    )

    assert canonical_parent_media_id("abc123_frame_0") == "abc123"
    assert canonical_parent_media_id("abc123_frame_12") == "abc123"
    assert canonical_parent_media_id("nomatch") == "nomatch"
    assert corpus_parent_media_id("any_id", {"parent_media_id": "parent-uuid"}) == "parent-uuid"
    assert corpus_parent_media_id("x_frame_2", None) == "x"
    exp = expand_media_ids_for_creator_lookup(["x_frame_2", "plain"])
    assert exp == ["x_frame_2", "x", "plain"]


def test_merge_modalities_one_row_per_parent():
    from app.models.similarity import MediaMatch, MatchType
    from app.models.media import ConfidenceLevel
    from app.services.poc_media_ids import merge_embedding_and_fingerprint_matches

    pid = "same-parent"
    emb = MediaMatch(
        media_id=pid,
        similarity_score=0.92,
        similarity_score_percent=92,
        match_type=MatchType.EMBEDDING,
        confidence_level=ConfidenceLevel.HIGH,
    )
    fp = MediaMatch(
        media_id=pid,
        similarity_score=1.0,
        similarity_score_percent=100,
        match_type=MatchType.FINGERPRINT,
        confidence_level=ConfidenceLevel.HIGH,
    )
    merged = merge_embedding_and_fingerprint_matches([emb], [fp])
    assert len(merged) == 1
    assert merged[0].match_type == MatchType.FINGERPRINT
    assert merged[0].similarity_score == 1.0


def test_merge_modalities_tie_prefers_fingerprint():
    from app.models.similarity import MediaMatch, MatchType
    from app.models.media import ConfidenceLevel
    from app.services.poc_media_ids import merge_embedding_and_fingerprint_matches

    pid = "p"
    emb = MediaMatch(
        media_id=pid,
        similarity_score=1.0,
        similarity_score_percent=100,
        match_type=MatchType.EMBEDDING,
        confidence_level=ConfidenceLevel.HIGH,
    )
    fp = MediaMatch(
        media_id=pid,
        similarity_score=1.0,
        similarity_score_percent=100,
        match_type=MatchType.FINGERPRINT,
        confidence_level=ConfidenceLevel.HIGH,
    )
    merged = merge_embedding_and_fingerprint_matches([emb], [fp])
    assert len(merged) == 1
    assert merged[0].match_type == MatchType.FINGERPRINT


def test_dedupe_media_matches_keeps_strongest_per_parent():
    from app.models.similarity import MediaMatch, MatchType
    from app.models.media import ConfidenceLevel
    from app.services.poc_media_ids import (
        canonical_parent_media_id,
        dedupe_media_matches_by_canonical_parent,
    )

    low = MediaMatch(
        media_id="corpus_frame_0",
        similarity_score=0.71,
        similarity_score_percent=71,
        match_type=MatchType.EMBEDDING,
        confidence_level=ConfidenceLevel.MEDIUM,
    )
    high = MediaMatch(
        media_id="corpus_frame_4",
        similarity_score=0.95,
        similarity_score_percent=95,
        match_type=MatchType.EMBEDDING,
        confidence_level=ConfidenceLevel.HIGH,
    )
    other = MediaMatch(
        media_id="other_vid_frame_1",
        similarity_score=0.99,
        similarity_score_percent=99,
        match_type=MatchType.EMBEDDING,
        confidence_level=ConfidenceLevel.HIGH,
    )
    out = dedupe_media_matches_by_canonical_parent([low, high, other])
    assert len(out) == 2
    by_parent = {canonical_parent_media_id(m.media_id): m for m in out}
    assert by_parent["corpus"].similarity_score == 0.95
    assert by_parent["other_vid"].similarity_score == 0.99


def test_compose_submission_royalty_free_sets_explicit_outcome():
    from app.models.similarity import MediaMatch, MatchType
    from app.models.media import ConfidenceLevel
    from app.services.poc_utils import OFFCHAIN_DEFAULT_POC_CONFIG, OUTCOME_ROYALTY_FREE

    m = MediaMatch(
        media_id="m1",
        similarity_score=0.5,
        similarity_score_percent=50,
        match_type=MatchType.EMBEDDING,
        confidence_level=ConfidenceLevel.LOW,
    )

    def resolve_creator(_ids):
        return None

    b = compose_submission_bundle(
        media_kind="image",
        media_type_code=1,
        matches=[m],
        video_analysis=None,
        poc_config=OFFCHAIN_DEFAULT_POC_CONFIG,
        resolve_creator=resolve_creator,
        royalty_free=True,
    )
    assert b["apply_explicit_outcome"] is True
    assert b["explicit_poc_outcome"] == OUTCOME_ROYALTY_FREE


    assert OFFCHAIN_DEFAULT_POC_CONFIG["video_threshold"] <= 100


def test_video_embedded_audio_only_path():
    from app.services.poc_video_types import VideoSimilarityAnalysis, decide_video_poc_for_chain

    analysis = VideoSimilarityAnalysis(
        matches=[],
        max_visual_similarity=0.5,
        best_visual_match_media_id="vis",
        audio_fingerprint_match=True,
        embedded_audio_similarity=1.0,
        best_audio_match_media_id="snd",
    )
    dec = decide_video_poc_for_chain(
        analysis,
        video_threshold_pct=95,
        audio_threshold_pct=95,
        visual_creator=None,
        audio_creator="0xabc",
    )
    assert dec.embedded_audio_only_derivative is True
    assert dec.original_creator == "0xabc"
    assert dec.highest_similarity_score_u64 == 100


def test_video_both_qualify_prefers_visual():
    from app.services.poc_video_types import VideoSimilarityAnalysis, decide_video_poc_for_chain

    analysis = VideoSimilarityAnalysis(
        matches=[],
        max_visual_similarity=0.99,
        best_visual_match_media_id="vis",
        audio_fingerprint_match=True,
        embedded_audio_similarity=1.0,
        best_audio_match_media_id="snd",
    )
    dec = decide_video_poc_for_chain(
        analysis,
        video_threshold_pct=95,
        audio_threshold_pct=95,
        visual_creator="0xVIDEO",
        audio_creator="0xAUDIO",
    )
    assert dec.embedded_audio_only_derivative is False
    assert dec.original_creator == "0xVIDEO"


def test_myso_post_active_poc_outcome():
    from app.services.myso_client import MySocialClient

    assert (
        MySocialClient.post_has_active_poc({"poc_outcome": "1", "revenue_redirect_to": None}) is True
    )
    blank = {
        "poc_outcome": "0",
        "revenue_redirect_to": None,
        "poc_badge_snapshot": None,
        "poc_badge_object_id": None,
    }
    assert MySocialClient.post_has_active_poc(blank) is False


def test_check_post_already_analyzed_allows_resubmit_after_overturn_clear():
    from app.services.myso_client import MySocialClient, POC_OUTCOME_NONE

    cleared_fields = {
        "poc_outcome": str(POC_OUTCOME_NONE),
        "revenue_redirect_to": None,
        "poc_badge_snapshot": None,
        "poc_badge_object_id": None,
        "poc_disputes_submitted": "1",
    }
    assert MySocialClient.post_has_active_poc(cleared_fields) is False
