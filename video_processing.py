import tempfile
import subprocess
from pathlib import Path
from typing import List

import embedding
import fingerprint


DEFAULT_KEYFRAME_RATE = 1  # frames per second


def extract_keyframes(video_path: str, rate: int = DEFAULT_KEYFRAME_RATE) -> List[str]:
    temp_dir = tempfile.mkdtemp()
    pattern = Path(temp_dir) / "frame_%04d.jpg"
    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-vf",
        f"fps={rate}",
        str(pattern),
        "-loglevel",
        "error",
    ]
    subprocess.run(cmd, check=True)
    return [str(p) for p in Path(temp_dir).glob("frame_*.jpg")]


def extract_audio(video_path: str) -> str:
    temp_file = tempfile.mktemp(suffix=".wav")
    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-map",
        "a",
        "-ac",
        "1",
        "-ar",
        "8000",
        temp_file,
        "-loglevel",
        "error",
    ]
    subprocess.run(cmd, check=True)
    return temp_file


def process_video(video_path: str):
    frames = extract_keyframes(video_path)
    audio_path = extract_audio(video_path)
    embeddings = [embedding.image_embedding(f) for f in frames]
    fp_hash = fingerprint.fingerprint_audio(audio_path)
    return embeddings, fp_hash
