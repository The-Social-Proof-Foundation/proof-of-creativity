"""Constellation verification and primary-hash helpers (aligned with README audio §)."""

import hashlib
import pickle
import wave
from pathlib import Path

import numpy as np
import pytest

from app.services.fingerprint import normalize_audio_fp_hash
from app.services.fingerprint import (
    audio_embedded_similarity_flags,
    find_verified_fingerprint_hits,
    fingerprint_audio_with_blob,
    unpickle_fingerprint_hashes,
    verify_constellation_against_blob,
)


def _write_mono_wav(path: Path, samples: np.ndarray, sr: int) -> None:
    samples = np.clip(samples.astype(np.float64), -1.0, 1.0)
    pcm = (samples * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _synthetic_constellation(n: int = 80) -> list:
    """Deterministic constellation tuples matching fingerprint.generate_hash_pairs shape."""
    rows = []
    for i in range(n):
        key = hashlib.sha1(f"p{i}|q{i}|{i % 40}".encode()).hexdigest()
        rows.append((key, i * 3, i % 12, (i + 7) % 12, 5 + (i % 20)))
    return rows


def test_primary_fp_hash_is_sha1_hex_length(tmp_path):
    sr = 8000
    tone = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr))
    path = tmp_path / "tone.wav"
    _write_mono_wav(path, tone, sr)
    fp_hash, _blob = fingerprint_audio_with_blob(str(path))
    assert isinstance(fp_hash, str) and len(fp_hash) == 40


def test_constellation_verify_identical_track():
    data = _synthetic_constellation()
    blob = pickle.dumps(data)
    qh = unpickle_fingerprint_hashes(blob)
    verdict = verify_constellation_against_blob(qh, blob)
    assert verdict.get("match") is True
    assert verdict.get("confidence", 0) > 0


def test_missing_constellation_data_returns_reason():
    data = _synthetic_constellation(10)
    qh = unpickle_fingerprint_hashes(pickle.dumps(data))
    verdict = verify_constellation_against_blob(qh, None)
    assert verdict.get("match") is False
    assert verdict.get("reason") == "missing_constellation_data"


def test_find_verified_skips_self():
    data = _synthetic_constellation()
    blob = pickle.dumps(data)
    qh = unpickle_fingerprint_hashes(blob)
    hits = find_verified_fingerprint_hits(
        query_hashes=qh,
        candidate_rows=[("upload-a", 0.0, blob)],
        query_media_id="upload-a",
    )
    assert hits == []


def test_find_verified_accepts_other_media():
    data = _synthetic_constellation()
    blob = pickle.dumps(data)
    qh = unpickle_fingerprint_hashes(blob)
    hits = find_verified_fingerprint_hits(
        query_hashes=qh,
        candidate_rows=[("corpus-other", 0.0, blob)],
        query_media_id="upload-a",
    )
    assert len(hits) == 1
    assert hits[0].corpus_media_id == "corpus-other"


def test_audio_embedded_similarity_flags_empty():
    ok, sim = audio_embedded_similarity_flags([])
    assert ok is False and sim == 0.0


def test_audio_embedded_similarity_flags_with_hit():
    data = _synthetic_constellation()
    blob = pickle.dumps(data)
    qh = unpickle_fingerprint_hashes(blob)
    hits = find_verified_fingerprint_hits(
        query_hashes=qh,
        candidate_rows=[("other", 0.0, blob)],
        query_media_id="me",
    )
    ok, sim = audio_embedded_similarity_flags(hits)
    assert ok is True
    assert sim > 0


def test_normalize_audio_fp_hash_accepts_uppercase_sha1():
    digest = hashlib.sha1(b"x").hexdigest()
    assert normalize_audio_fp_hash(digest.upper()) == digest


def test_normalize_audio_fp_hash_rejects_short_string():
    with pytest.raises(ValueError, match="40 hex"):
        normalize_audio_fp_hash("80055d942e")


def test_normalize_audio_fp_hash_rejects_non_hex():
    with pytest.raises(ValueError, match="hexadecimal"):
        normalize_audio_fp_hash("g" * 40)
