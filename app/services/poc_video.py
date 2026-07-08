"""Video modality similarity: frame embeddings + embedded audio fingerprints."""

from __future__ import annotations

from typing import List, Optional

from app.models.media import ConfidenceLevel
from app.models.similarity import MediaMatch, MatchType
from app.services.poc_media_ids import (
    corpus_parent_media_id,
    dedupe_media_matches_by_canonical_parent,
    merge_embedding_and_fingerprint_matches,
)
from app.services.poc_utils import similarity_float_to_u64_percent
from app.services.poc_video_types import VideoSimilarityAnalysis
from app.services.video_processing import process_video
from app.services.fingerprint import (
    audio_embedded_similarity_flags,
    find_verified_fingerprint_hits,
    unpickle_fingerprint_hashes,
)
from app.services.media_similarity import corpus_search_kwargs, insert_kwargs_from_context
from app.core.database import (
    insert_embedding,
    insert_fingerprint,
    insert_similarity_match,
    search_embedding_with_timescale_ai,
    search_fingerprint_rows,
)


def analyze_video_similarity(temp_file_path: str, query_media_id: str) -> VideoSimilarityAnalysis:
    """
    Run frame embeddings + audio fingerprint paths without merging modality decisions.
    """
    frame_embeddings, audio_fp_hash, audio_fp_blob = process_video(temp_file_path)
    matches: List[MediaMatch] = []
    audio_similarity = 0.0
    audio_hit = False

    search_kwargs = corpus_search_kwargs()
    insert_kwargs = insert_kwargs_from_context()

    for idx, frame_embedding in enumerate(frame_embeddings):
        similar_embeddings = search_embedding_with_timescale_ai(
            vector=frame_embedding,
            top_k=5,
            kind_filter="video_frame",
            similarity_threshold=0.7,
            **search_kwargs,
        )
        frame_media_id = f"{query_media_id}_frame_{idx}"
        insert_embedding(
            frame_media_id,
            "video_frame",
            frame_embedding,
            {"parent_media_id": query_media_id, "frame_index": idx},
            **insert_kwargs,
        )
        for match in similar_embeddings:
            match_media_id, kind, metadata, uploaded_at, similarity_score, distance = match
            if match_media_id.startswith(query_media_id):
                continue
            sim = float(similarity_score)
            confidence: ConfidenceLevel
            if sim > 0.9:
                confidence = ConfidenceLevel.HIGH
            elif sim > 0.7:
                confidence = ConfidenceLevel.MEDIUM
            else:
                confidence = ConfidenceLevel.LOW
            insert_similarity_match(
                query_media_id=query_media_id,
                match_media_id=match_media_id,
                match_type="embedding",
                similarity_score=sim,
                confidence_level=confidence.value,
                match_details={"embedding_type": "clip", "kind": kind, "frame_index": idx},
            )
            meta_dict = metadata if isinstance(metadata, dict) else None
            matches.append(
                MediaMatch(
                    media_id=corpus_parent_media_id(match_media_id, meta_dict),
                    similarity_score=sim,
                    similarity_score_percent=similarity_float_to_u64_percent(sim),
                    match_type=MatchType.EMBEDDING,
                    confidence_level=confidence,
                )
            )

    if audio_fp_hash:
        candidate_rows = search_fingerprint_rows(audio_fp_hash, **search_kwargs)
        query_hashes = unpickle_fingerprint_hashes(audio_fp_blob)
        verified_hits = find_verified_fingerprint_hits(
            query_hashes=query_hashes,
            candidate_rows=candidate_rows,
            query_media_id=query_media_id,
        )
        insert_fingerprint(audio_fp_hash, query_media_id, 0.0, audio_fp_blob, **insert_kwargs)
        audio_hit, audio_similarity = audio_embedded_similarity_flags(verified_hits)
        for hit in verified_hits:
            v = hit.verdict
            match_media_id = hit.corpus_media_id
            insert_similarity_match(
                query_media_id=query_media_id,
                match_media_id=match_media_id,
                match_type="fingerprint",
                similarity_score=float(hit.similarity_score),
                confidence_level=hit.confidence_level.value,
                match_details={
                    "fingerprint_hash": audio_fp_hash,
                    "offset": v.get("offset"),
                    "matching_hashes": v.get("matching_hashes"),
                    "offset_matches": v.get("offset_matches"),
                    "constellation_verified": True,
                },
            )
            matches.append(
                MediaMatch(
                    media_id=corpus_parent_media_id(match_media_id, None),
                    similarity_score=float(hit.similarity_score),
                    similarity_score_percent=similarity_float_to_u64_percent(float(hit.similarity_score)),
                    match_type=MatchType.FINGERPRINT,
                    confidence_level=hit.confidence_level,
                )
            )

    embedding_matches = [m for m in matches if m.match_type == MatchType.EMBEDDING]
    fingerprint_matches = [m for m in matches if m.match_type == MatchType.FINGERPRINT]
    deduped_emb = dedupe_media_matches_by_canonical_parent(embedding_matches)
    deduped_fp = dedupe_media_matches_by_canonical_parent(fingerprint_matches)
    merged = merge_embedding_and_fingerprint_matches(deduped_emb, deduped_fp)

    max_visual = max((m.similarity_score for m in deduped_emb), default=0.0)
    best_visual_mid: Optional[str] = None
    if deduped_emb:
        best_emb = max(deduped_emb, key=lambda x: x.similarity_score)
        best_visual_mid = best_emb.media_id

    best_audio_mid: Optional[str] = None
    if deduped_fp:
        best_fp = max(deduped_fp, key=lambda x: x.similarity_score)
        best_audio_mid = best_fp.media_id

    return VideoSimilarityAnalysis(
        matches=merged,
        max_visual_similarity=max_visual,
        best_visual_match_media_id=best_visual_mid,
        audio_fingerprint_match=audio_hit,
        embedded_audio_similarity=audio_similarity,
        best_audio_match_media_id=best_audio_mid,
    )
