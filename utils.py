import uuid
from pathlib import Path
import tempfile


def save_temp_upload(upload_file) -> str:
    suffix = Path(upload_file.filename).suffix
    path = Path(tempfile.mktemp(suffix=suffix))
    with path.open("wb") as f:
        f.write(upload_file.file.read())
    upload_file.file.seek(0)
    return str(path)


def new_media_id() -> str:
    return str(uuid.uuid4())
