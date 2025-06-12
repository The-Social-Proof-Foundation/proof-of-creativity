import numpy as np
import librosa
import hashlib


def fingerprint_audio(file_path: str) -> str:
    y, sr = librosa.load(file_path, sr=8000, mono=True)
    # Simple fingerprint using spectrogram hash
    S = np.abs(librosa.stft(y))
    S = librosa.feature.melspectrogram(S=S, sr=sr)
    fp = hashlib.sha1(S.tobytes()).hexdigest()
    return fp
