"""Unit tests for discovery confidence thresholds."""

import os
from unittest.mock import patch

from app.discovery.confidence import (
    cold_start_work_confidence,
    creator_confidence_threshold,
    passes_off_network_thresholds,
    work_confidence_threshold,
)


def test_default_thresholds():
    with patch.dict(os.environ, {}, clear=False):
        assert work_confidence_threshold() == 0.95
        assert creator_confidence_threshold() == 0.85
        assert cold_start_work_confidence() == 0.0


def test_passes_off_network_requires_both():
    assert passes_off_network_thresholds(
        identity_hash="0xabc",
        creator_confidence=0.9,
        work_confidence=0.96,
    )
    assert not passes_off_network_thresholds(
        identity_hash="0xabc",
        creator_confidence=0.9,
        work_confidence=0.5,
    )
    assert not passes_off_network_thresholds(
        identity_hash="0xabc",
        creator_confidence=0.5,
        work_confidence=0.99,
    )
    assert not passes_off_network_thresholds(
        identity_hash=None,
        creator_confidence=1.0,
        work_confidence=1.0,
    )


def test_env_overrides():
    with patch.dict(
        os.environ,
        {
            "DISCOVERY_WORK_CONFIDENCE_THRESHOLD": "0.80",
            "DISCOVERY_X_HANDLE_CONFIDENCE_THRESHOLD": "0.70",
            "DISCOVERY_COLD_START_WORK_CONFIDENCE": "0.1",
        },
        clear=False,
    ):
        assert work_confidence_threshold() == 0.80
        assert creator_confidence_threshold() == 0.70
        assert cold_start_work_confidence() == 0.1
