"""Shared off-network discovery confidence threshold helpers."""

from __future__ import annotations

import os


def work_confidence_threshold() -> float:
    return float(
        os.getenv(
            "POC_WORK_CONFIDENCE_THRESHOLD",
            os.getenv("DISCOVERY_WORK_CONFIDENCE_THRESHOLD", "0.95"),
        )
    )


def creator_confidence_threshold() -> float:
    return float(
        os.getenv(
            "POC_X_HANDLE_CONFIDENCE_THRESHOLD",
            os.getenv("DISCOVERY_X_HANDLE_CONFIDENCE_THRESHOLD", "0.85"),
        )
    )


def cold_start_work_confidence() -> float:
    return float(
        os.getenv(
            "POC_COLD_START_WORK_CONFIDENCE",
            os.getenv("DISCOVERY_COLD_START_WORK_CONFIDENCE", "0.0"),
        )
    )


def passes_off_network_thresholds(
    *,
    identity_hash: str | None,
    creator_confidence: float,
    work_confidence: float,
) -> bool:
    if not identity_hash:
        return False
    return (
        creator_confidence >= creator_confidence_threshold()
        and work_confidence >= work_confidence_threshold()
    )
