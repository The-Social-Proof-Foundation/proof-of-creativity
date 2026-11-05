"""Unit tests for similarity scoring helpers."""

from app.services.analysis.scoring import similarity_float_to_u64_percent


def test_similarity_scaling():
    assert similarity_float_to_u64_percent(0.0) == 0
    assert similarity_float_to_u64_percent(0.949) == 95
    assert similarity_float_to_u64_percent(0.995) == 100
