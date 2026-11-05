"""Private-source validation for standalone catalog audio resolution."""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.core.utils import cleanup_temp_file, new_media_id
from app.services.analysis.media_fetcher import download_media
from app.services.media_similarity import detect_audio_similarity
from app.services.poc_utils import similarity_float_to_u64_percent


@dataclass(frozen=True)
class StandaloneAudioAnalysis:
    path: str
    media_id: str
    highest_similarity_bps: int
    matched_media_ids: tuple[str, ...]


def _hex_bytes(value: str) -> bytes:
    return bytes.fromhex(str(value or "").lower().removeprefix("0x"))


def _normalized_pcm(path: str) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as tmp:
        pcm_path = tmp.name
    try:
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", path,
                "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", pcm_path,
            ],
            check=True,
            timeout=180,
        )
        pcm = Path(pcm_path).read_bytes()
        if not pcm:
            raise ValueError("standalone audio decoded to empty PCM")
        return pcm
    finally:
        Path(pcm_path).unlink(missing_ok=True)


def verify_audio_commitments(
    path: str,
    *,
    content_commitment_hex: str,
    observed_fingerprint_commitment_hex: str,
) -> None:
    content = Path(path).read_bytes()
    if not content:
        raise ValueError("standalone audio source is empty")
    expected_content = _hex_bytes(content_commitment_hex)
    if hashlib.sha256(content).digest() != expected_content:
        raise ValueError("standalone audio content commitment mismatch")
    expected = _hex_bytes(observed_fingerprint_commitment_hex)
    pcm = _normalized_pcm(path)
    prefixes = (
        b"poc:observed_fp:v1:audio_pcm:",
        b"dripdrop:observed_fp:v1:audio_pcm:",
    )
    if not any(hashlib.sha256(prefix + pcm).digest() == expected for prefix in prefixes):
        raise ValueError("standalone audio fingerprint commitment mismatch")


async def fetch_and_analyze_standalone_audio(
    *,
    source_url: str,
    content_commitment_hex: str,
    observed_fingerprint_commitment_hex: str,
) -> StandaloneAudioAnalysis:
    path, _content_type = await download_media(source_url)
    try:
        verify_audio_commitments(
            path,
            content_commitment_hex=content_commitment_hex,
            observed_fingerprint_commitment_hex=observed_fingerprint_commitment_hex,
        )
        media_id = new_media_id()
        matches = await detect_audio_similarity(path, media_id)
        highest = max(
            (similarity_float_to_u64_percent(match.similarity_score) * 100 for match in matches),
            default=0,
        )
        return StandaloneAudioAnalysis(
            path=path,
            media_id=media_id,
            highest_similarity_bps=highest,
            matched_media_ids=tuple(match.media_id for match in matches),
        )
    finally:
        cleanup_temp_file(path)


def requires_relationship_review(analysis: StandaloneAudioAnalysis) -> bool:
    threshold_bps = int(os.getenv("POC_AUDIO_RELATIONSHIP_REVIEW_BPS", "8500"))
    return analysis.highest_similarity_bps >= threshold_bps
