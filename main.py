from fastapi import FastAPI, UploadFile, File, HTTPException
import mimetypes
import os

import embedding
import fingerprint
import video_processing
from storage import StorageClient
from db import insert_embedding, search_embedding, insert_fingerprint, search_fingerprint
from utils import save_temp_upload, new_media_id

app = FastAPI()
storage_client = StorageClient()


def detect_image(path: str, media_id: str):
    emb = embedding.image_embedding(path)
    matches = search_embedding(emb)
    insert_embedding(media_id, "image", emb, {})
    return matches


def detect_audio(path: str, media_id: str):
    fp_hash = fingerprint.fingerprint_audio(path)
    matches = search_fingerprint(fp_hash)
    insert_fingerprint(fp_hash, media_id, 0)
    return matches


def detect_video(path: str, media_id: str):
    embeddings, fp_hash = video_processing.process_video(path)
    matches = []
    for idx, emb in enumerate(embeddings):
        matches.extend(search_embedding(emb))
        insert_embedding(f"{media_id}_{idx}", "video_frame", emb, {"parent": media_id})
    matches.extend(search_fingerprint(fp_hash))
    insert_fingerprint(fp_hash, media_id, 0)
    return matches


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    media_id = new_media_id()
    path = save_temp_upload(file)
    mtype = file.content_type or mimetypes.guess_type(file.filename)[0]

    if mtype and mtype.startswith("image"):
        matches = detect_image(path, media_id)
    elif mtype and mtype.startswith("audio"):
        matches = detect_audio(path, media_id)
    elif mtype and mtype.startswith("video"):
        matches = detect_video(path, media_id)
    else:
        raise HTTPException(status_code=400, detail="Unsupported media type")

    with open(path, "rb") as f:
        uri = storage_client.upload(file.filename, f)

    return {"media_id": media_id, "matches": matches, "uri": uri}
