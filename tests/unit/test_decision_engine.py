"""Unit tests for decision engine."""

from unittest.mock import MagicMock

from app.models.media import ConfidenceLevel
from app.models.similarity import MatchType, MediaMatch
from app.services.analysis.pipeline import AnalysisResult
from app.services.decision_engine import DecisionEngine
from app.services.poc_utils import DERIVATIVE_TARGET_ESCROW, DERIVATIVE_TARGET_WALLET


def test_original_when_no_matches(monkeypatch):
    engine = DecisionEngine()
    monkeypatch.setattr(engine, "_load_config", lambda _n: {"image_threshold": 95, "video_threshold": 95, "audio_threshold": 95})
    analysis = AnalysisResult(
        network="localnet",
        post_id="0x1",
        media_url="https://x/y.jpg",
        media_index=0,
        media_type=1,
        media_id="mid",
        matches=[],
        highest_similarity_u64=0,
    )
    submission = engine.build_submission({}, analysis)
    assert submission.original_creator is None
    assert submission.highest_similarity_score == 0


def test_off_network_forces_escrow(monkeypatch):
    engine = DecisionEngine()
    monkeypatch.setattr(engine, "_load_config", lambda _n: {"image_threshold": 95, "video_threshold": 95, "audio_threshold": 95})
    match = MediaMatch(
        media_id="m2",
        match_type=MatchType.PERCEPTUAL_HASH,
        similarity_score=0.99,
        confidence_level=ConfidenceLevel.HIGH,
        match_details={"identity_hash": "abc123"},
    )
    analysis = AnalysisResult(
        network="localnet",
        post_id="0x2",
        media_url="https://x/y.jpg",
        media_index=0,
        media_type=1,
        media_id="mid",
        matches=[match],
        highest_similarity_u64=99,
        off_network=True,
        identity_hash="abc123",
    )
    submission = engine.build_submission({}, analysis)
    assert submission.derivative_redirection_target == DERIVATIVE_TARGET_ESCROW
