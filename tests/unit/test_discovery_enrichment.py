"""Unit tests for match detail enrichment."""

from unittest.mock import MagicMock, patch

from app.models.media import ConfidenceLevel
from app.models.similarity import MatchType
from app.services.media_similarity import _enrich_match_details


def test_enrich_match_details_includes_creator_x_handle():
    repo = MagicMock()
    repo.get_embedding_provenance.return_value = {
        "corpus_scope": "discovered",
        "discovery_asset_id": "asset-1",
        "metadata": {
            "identity_hash": "0xabc",
            "creator_x_handle": "creatorname",
            "creator_confidence": 0.9,
            "work_confidence": 0.97,
        },
    }
    with patch("app.services.media_similarity._provenance_repo", repo):
        details = _enrich_match_details("disc_123", {"hash_type": "dhash"})
    assert details["corpus_scope"] == "discovered"
    assert details["creator_x_handle"] == "creatorname"
    assert details["identity_hash"] == "0xabc"
    assert details["work_confidence"] == 0.97
