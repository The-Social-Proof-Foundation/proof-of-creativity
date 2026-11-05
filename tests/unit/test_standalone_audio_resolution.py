from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from app.services.standalone_audio_resolution import verify_audio_commitments


def test_verify_audio_commitments_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "tone.m4a"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
            "-i", "sine=frequency=440:duration=0.2", "-c:a", "aac", "-ar", "48000",
            str(source),
        ],
        check=True,
    )
    pcm = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(source),
            "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-",
        ],
        check=True,
        capture_output=True,
    ).stdout
    verify_audio_commitments(
        str(source),
        content_commitment_hex=hashlib.sha256(source.read_bytes()).hexdigest(),
        observed_fingerprint_commitment_hex=hashlib.sha256(
            b"dripdrop:observed_fp:v1:audio_pcm:" + pcm
        ).hexdigest(),
    )
    verify_audio_commitments(
        str(source),
        content_commitment_hex=hashlib.sha256(source.read_bytes()).hexdigest(),
        observed_fingerprint_commitment_hex=hashlib.sha256(
            b"poc:observed_fp:v1:audio_pcm:" + pcm
        ).hexdigest(),
    )


def test_verify_audio_commitments_rejects_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "bad.m4a"
    source.write_bytes(b"not audio")
    with pytest.raises(ValueError, match="content commitment mismatch"):
        verify_audio_commitments(
            str(source),
            content_commitment_hex="00" * 32,
            observed_fingerprint_commitment_hex="00" * 32,
        )
