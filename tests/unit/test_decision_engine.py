"""Unit tests for decision engine."""

from unittest.mock import MagicMock

from app.models.media import ConfidenceLevel
from app.models.similarity import MatchType, MediaMatch
from app.services.analysis.pipeline import AnalysisResult
from app.services.decision_engine import DecisionEngine
from app.services.poc_utils import DERIVATIVE_TARGET_ESCROW, DERIVATIVE_TARGET_WALLET


def _cfg():
    return {"image_threshold": 95, "video_threshold": 95, "audio_threshold": 95}


def test_original_when_no_matches(monkeypatch):
    engine = DecisionEngine()
    monkeypatch.setattr(engine, "_load_config", lambda _n: _cfg())
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
    monkeypatch.setattr(engine, "_load_config", lambda _n: _cfg())
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
        creator_confidence=0.9,
    )
    submission = engine.build_submission({}, analysis)
    assert submission.derivative_redirection_target == DERIVATIVE_TARGET_ESCROW


def test_self_match_bypasses_derivative(monkeypatch):
    engine = DecisionEngine()
    monkeypatch.setattr(engine, "_load_config", lambda _n: _cfg())
    owner = "0xAbc123"
    match = MediaMatch(
        media_id="m1",
        match_type=MatchType.PERCEPTUAL_HASH,
        similarity_score=1.0,
        confidence_level=ConfidenceLevel.HIGH,
    )
    analysis = AnalysisResult(
        network="localnet",
        post_id="0xpost",
        media_url="https://x/y.jpg",
        media_index=0,
        media_type=1,
        media_id="mid",
        matches=[match],
        highest_similarity_u64=100,
        original_creator=owner,
        reasoning="top match",
    )
    submission = engine.build_submission({"creator_address": owner}, analysis)
    assert submission.original_creator is None
    assert submission.highest_similarity_score == 100
    assert submission.derivative_redirection_target == DERIVATIVE_TARGET_WALLET
    assert "Self-match" in submission.reasoning


def test_different_creator_stays_derivative(monkeypatch):
    engine = DecisionEngine()
    monkeypatch.setattr(engine, "_load_config", lambda _n: _cfg())
    owner = "0xowner"
    other = "0xother"
    match = MediaMatch(
        media_id="m1",
        match_type=MatchType.PERCEPTUAL_HASH,
        similarity_score=1.0,
        confidence_level=ConfidenceLevel.HIGH,
    )
    analysis = AnalysisResult(
        network="localnet",
        post_id="0xpost",
        media_url="https://x/y.jpg",
        media_index=0,
        media_type=1,
        media_id="mid",
        matches=[match],
        highest_similarity_u64=100,
        original_creator=other,
        reasoning="top match",
    )
    submission = engine.build_submission({"creator_address": owner}, analysis)
    assert submission.original_creator == other
    assert submission.highest_similarity_score == 100
    assert submission.derivative_redirection_target == DERIVATIVE_TARGET_WALLET
