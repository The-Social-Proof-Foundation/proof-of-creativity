import io
import os
import numpy as np
import soundfile as sf
from utils import new_media_id, save_temp_upload
import fingerprint

def test_new_media_id_unique():
    ids = {new_media_id() for _ in range(5)}
    assert len(ids) == 5

def test_fingerprint_audio(tmp_path):
    sr = 8000
    tone = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr))
    file_path = tmp_path / "tone.wav"
    sf.write(file_path, tone, sr)
    fp = fingerprint.fingerprint_audio(str(file_path))
    assert isinstance(fp, str) and len(fp) == 40


def test_save_temp_upload(tmp_path):
    class DummyUpload:
        def __init__(self, name, data):
            self.filename = name
            self.file = io.BytesIO(data)

    import io
    upload = DummyUpload("test.txt", b"hello")
    path = save_temp_upload(upload)
    assert os.path.exists(path)
    os.remove(path)
