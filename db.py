import os
import psycopg2
from typing import List, Any, Dict
from psycopg2 import extras

DB_DSN = os.getenv("DB_DSN", "postgresql://user:password@localhost:5432/proof")


def get_conn():
    return psycopg2.connect(DB_DSN)


def insert_embedding(media_id: str, kind: str, embedding: List[float], metadata: Dict[str, Any]):
    sql = """
    INSERT INTO media_embeddings (media_id, kind, embedding, metadata)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (media_id) DO NOTHING
    """
    conn = get_conn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(sql, (media_id, kind, embedding, extras.Json(metadata)))
    finally:
        conn.close()


def insert_fingerprint(fp_hash: str, media_id: str, offset: float):
    sql = """
    INSERT INTO audio_fingerprints (fp_hash, media_id, offset)
    VALUES (%s, %s, %s)
    ON CONFLICT DO NOTHING
    """
    conn = get_conn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(sql, (fp_hash, media_id, offset))
    finally:
        conn.close()


def search_fingerprint(fp_hash: str):
    sql = "SELECT media_id, offset FROM audio_fingerprints WHERE fp_hash = %s"
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (fp_hash,))
            return cur.fetchall()
    finally:
        conn.close()


def search_embedding(vector: List[float], top_k: int = 5):
    sql = """
    SELECT media_id, kind, metadata, embedding <-> %s AS dist
    FROM media_embeddings
    ORDER BY dist
    LIMIT %s
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (vector, top_k))
            return cur.fetchall()
    finally:
        conn.close()
