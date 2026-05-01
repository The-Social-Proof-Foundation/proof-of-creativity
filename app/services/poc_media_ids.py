"""Normalize oracle media_id values for PoC creator lookups (video frames, etc.)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.models.similarity import MatchType, MediaMatch

# Embeddings for video frames use "{parent}_frame_{idx}" (see poc_video.py).
_FRAME_SUFFIX = re.compile(r"_frame_\d+$")


def canonical_parent_media_id(media_id: str) -> str:
    """Strip synthetic video-frame suffix so lookups hit the parent's media_files row."""
    if not media_id:
        return media_id
    return _FRAME_SUFFIX.sub("", media_id)


def corpus_parent_media_id(
    media_id: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Stable parent id for a corpus embedding/fingerprint row.

    Video frame rows stored in media_embeddings include parent_media_id in metadata; using it
    avoids inflated match counts when media_id strings are not consistently suffixed with
    _frame_<n> (and dedupes all frames of the same upload to one source).
    """
    if not media_id:
        return media_id
    mid = str(media_id).strip()
    if metadata and isinstance(metadata, dict):
        p = metadata.get("parent_media_id")
        if p is not None and str(p).strip():
            return str(p).strip()
    return canonical_parent_media_id(mid)


def expand_media_ids_for_creator_lookup(media_ids: List[str]) -> List[str]:
    """Return de-duplicated ids including canonical parents for frame-style ids."""
    ordered: List[str] = []
    seen: set[str] = set()
    for raw in media_ids:
        if not raw or not str(raw).strip():
            continue
        mid = str(raw).strip()
        for candidate in (mid, canonical_parent_media_id(mid)):
            if candidate and candidate not in seen:
                seen.add(candidate)
                ordered.append(candidate)
    return ordered


def dedupe_media_matches_by_canonical_parent(matches: List[MediaMatch]) -> List[MediaMatch]:
    """
    Collapse vector hits that only differ by frame index (e.g. vid_frame_0 vs vid_frame_4)
    into one row per logical corpus upload, keeping the strongest similarity.

    Without this, proof_data.matches_found counts frame-level neighbors, which inflates
    counts versus "how many distinct source assets matched."
    """
    by_parent: Dict[str, MediaMatch] = {}
    for m in matches:
        key = canonical_parent_media_id(m.media_id)
        if not key:
            continue
        prev = by_parent.get(key)
        if prev is None or m.similarity_score > prev.similarity_score:
            by_parent[key] = m
    return list(by_parent.values())


def merge_embedding_and_fingerprint_matches(
    embedding_matches: List[MediaMatch],
    fingerprint_matches: List[MediaMatch],
) -> List[MediaMatch]:
    """
    One match per corpus parent across modalities (same source hitting both CLIP and audio
    fingerprint should not double matches_found).
    """
    def _is_fp(x: MediaMatch) -> bool:
        return x.match_type == MatchType.FINGERPRINT or x.match_type == "fingerprint"

    best: Dict[str, MediaMatch] = {}
    for m in embedding_matches + fingerprint_matches:
        mid = m.media_id
        if not mid:
            continue
        prev = best.get(mid)
        if prev is None:
            best[mid] = m
            continue
        if m.similarity_score > prev.similarity_score:
            best[mid] = m
        elif m.similarity_score == prev.similarity_score and _is_fp(m) and not _is_fp(prev):
            best[mid] = m
    return list(best.values())
