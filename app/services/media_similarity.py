"""Image and audio similarity entrypoints shared by REST + background upload pipeline."""

from __future__ import annotations

import structlog
from typing import List

from app.services import image_hash
from app.services.embedding import image_embedding
from app.services.fingerprint import (
    fingerprint_audio_with_blob,
    find_verified_fingerprint_hits,
    unpickle_fingerprint_hashes,
)
from app.core.database import (
    insert_embedding,
    insert_fingerprint,
    insert_image_hashes,
    insert_similarity_match,
    search_embedding_with_timescale_ai,
    search_fingerprint_rows,
    search_similar_image_hashes,
)
from app.models.media import ConfidenceLevel
from app.models.similarity import MatchType, MediaMatch
from app.services.poc_utils import similarity_float_to_u64_percent

logger = structlog.get_logger()


async def detect_image_similarity(file_path: str, media_id: str) -> List[MediaMatch]:
    """Process image: perceptual hash + optional CLIP embedding search."""
    try:
        logger.info("Processing image for similarity detection", media_id=media_id)
        image_hashes = image_hash.generate_image_hashes(file_path)
        hash_matches = search_similar_image_hashes(image_hashes, exclude_media_id=media_id)
        insert_image_hashes(media_id, image_hashes)
        matches: List[MediaMatch] = []

        for hash_match in hash_matches:
            match_media_id = hash_match[0]
            stored_hashes = {
                "dhash": hash_match[1],
                "phash": hash_match[2],
                "ahash": hash_match[3],
                "dhash_16": hash_match[4],
                "phash_16": hash_match[5],
            }
            hash_similarity, match_type = image_hash.calculate_hash_similarity(image_hashes, stored_hashes)
            if hash_similarity >= 0.99:
                confidence = ConfidenceLevel.HIGH
                insert_similarity_match(
                    query_media_id=media_id,
                    match_media_id=match_media_id,
                    match_type="perceptual_hash",
                    similarity_score=float(hash_similarity),
                    confidence_level=confidence.value,
                    match_details={
                        "hash_type": match_type,
                        "match_category": "exact_duplicate",
                        "hash_similarity": hash_similarity,
                    },
                )
                matches.append(
                    MediaMatch(
                        media_id=match_media_id,
                        similarity_score=float(hash_similarity),
                        similarity_score_percent=similarity_float_to_u64_percent(float(hash_similarity)),
                        match_type=MatchType.PERCEPTUAL_HASH,
                        confidence_level=confidence,
                    )
                )

        if not matches:
            logger.debug("No perceptual hash duplicates, checking CLIP", media_id=media_id)
            embedding_vector = image_embedding(file_path)
            duplicate_threshold = 0.98
            similar_threshold = 0.90
            duplicate_embeddings = search_embedding_with_timescale_ai(
                vector=embedding_vector,
                top_k=10,
                kind_filter="image",
                similarity_threshold=duplicate_threshold,
            )
            similar_embeddings = (
                search_embedding_with_timescale_ai(
                    vector=embedding_vector,
                    top_k=10,
                    kind_filter="image",
                    similarity_threshold=similar_threshold,
                )
                if not duplicate_embeddings
                else []
            )
            insert_embedding(media_id, "image", embedding_vector, {})
            all_embeddings = duplicate_embeddings + similar_embeddings
            for match in all_embeddings:
                match_media_id, kind, metadata, uploaded_at, similarity_score, distance = match
                if match_media_id == media_id:
                    continue
                if similarity_score >= 0.99:
                    confidence = ConfidenceLevel.HIGH
                    match_category = "semantic_duplicate"
                elif similarity_score >= 0.98:
                    confidence = ConfidenceLevel.HIGH
                    match_category = "semantic_near_duplicate"
                elif similarity_score >= 0.95:
                    confidence = ConfidenceLevel.MEDIUM
                    match_category = "similar_content"
                elif similarity_score >= 0.90:
                    confidence = ConfidenceLevel.MEDIUM
                    match_category = "related_content"
                else:
                    confidence = ConfidenceLevel.LOW
                    match_category = "loosely_related"
                insert_similarity_match(
                    query_media_id=media_id,
                    match_media_id=match_media_id,
                    match_type="embedding",
                    similarity_score=float(similarity_score),
                    confidence_level=confidence.value,
                    match_details={
                        "embedding_type": "clip",
                        "kind": kind,
                        "match_category": match_category,
                        "duplicate_threshold": duplicate_threshold,
                        "similar_threshold": similar_threshold,
                    },
                )
                matches.append(
                    MediaMatch(
                        media_id=match_media_id,
                        similarity_score=float(similarity_score),
                        similarity_score_percent=similarity_float_to_u64_percent(float(similarity_score)),
                        match_type=MatchType.EMBEDDING,
                        confidence_level=confidence,
                    )
                )
        else:
            try:
                embedding_vector = image_embedding(file_path)
                insert_embedding(media_id, "image", embedding_vector, {})
            except Exception as exc:
                logger.warning(
                    "CLIP embedding storage skipped after hash match",
                    media_id=media_id,
                    error=str(exc),
                )

        return matches

    except Exception as e:
        logger.error("Failed to detect image similarity", media_id=media_id, error=str(e))
        raise RuntimeError(f"Image similarity detection failed: {e}") from e


async def detect_audio_similarity(file_path: str, media_id: str) -> List[MediaMatch]:
    """Process audio: fp_hash candidates + constellation verification (offset clustering)."""
    try:
        fp_hash, fp_blob = fingerprint_audio_with_blob(file_path)
        candidate_rows = search_fingerprint_rows(fp_hash)
        query_hashes = unpickle_fingerprint_hashes(fp_blob)
        verified_hits = find_verified_fingerprint_hits(
            query_hashes=query_hashes,
            candidate_rows=candidate_rows,
            query_media_id=media_id,
        )
        insert_fingerprint(fp_hash, media_id, 0.0, fp_blob)
        matches: List[MediaMatch] = []
        for hit in verified_hits:
            v = hit.verdict
            insert_similarity_match(
                query_media_id=media_id,
                match_media_id=hit.corpus_media_id,
                match_type="fingerprint",
                similarity_score=float(hit.similarity_score),
                confidence_level=hit.confidence_level.value,
                match_details={
                    "fingerprint_hash": fp_hash,
                    "offset": v.get("offset"),
                    "matching_hashes": v.get("matching_hashes"),
                    "offset_matches": v.get("offset_matches"),
                    "constellation_verified": True,
                },
            )
            matches.append(
                MediaMatch(
                    media_id=hit.corpus_media_id,
                    similarity_score=float(hit.similarity_score),
                    similarity_score_percent=similarity_float_to_u64_percent(float(hit.similarity_score)),
                    match_type=MatchType.FINGERPRINT,
                    confidence_level=hit.confidence_level,
                )
            )
        return matches
    except Exception as e:
        logger.error("Audio similarity failed", media_id=media_id, error=str(e))
        raise RuntimeError(f"Error processing audio: {e}") from e
