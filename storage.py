import os
from typing import BinaryIO

from google.cloud import storage
import requests

import config


class StorageClient:
    def __init__(self):
        self.gcs_client = None
        if config.USE_GCS:
            self.gcs_client = storage.Client()

    def upload(self, filename: str, fileobj: BinaryIO) -> str:
        if config.USE_GCS and self.gcs_client:
            bucket = self.gcs_client.bucket(config.GCS_BUCKET_NAME)
            blob = bucket.blob(filename)
            blob.upload_from_file(fileobj)
            return f"gs://{config.GCS_BUCKET_NAME}/{filename}"
        elif config.USE_WALRUS:
            resp = requests.put(
                f"{config.WALRUS_ENDPOINT}/upload/{filename}", data=fileobj
            )
            resp.raise_for_status()
            return resp.json().get("cid")
        else:
            raise RuntimeError("No storage backend configured")
