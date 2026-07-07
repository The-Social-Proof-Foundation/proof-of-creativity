# 🧠 Proof of Creativity: Scalable Media Attribution Architecture

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Timescale](https://img.shields.io/badge/Timescale-Vector%20AI-orange.svg)](https://timescale.com/ai)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready system for detecting original vs derivative media content using advanced AI fingerprinting, CLIP embeddings, and Timescale Vector AI for lightning-fast similarity search.

## 🎯 Overview

This system enables real-time detection of:
- **Original vs derivative media** (images, audio, video)
- **Content attribution** and ownership tracking
- **Blockchain-ready** proof of creativity
- **Multi-modal similarity** with confidence scoring

### 🚀 Key Features

- **🔍 Advanced Similarity Detection**
  - CLIP embeddings for visual content
  - Shazam-style audio fingerprinting 
  - Video frame extraction and matching
  - Timescale Vector AI for sub-second searches

- **⚡ High Performance**
  - GPU/MPS acceleration support
  - Connection pooling and batch processing
  - HNSW vector indexes for optimal search
  - Scalable to millions of media files

- **🛡️ Production Ready**
  - Comprehensive error handling and logging
  - File validation and security measures
  - Health monitoring and metrics
  - Multi-backend storage (GCS/Walrus)

- **🔗 Blockchain Integration**
  - Attribution tracking and proof storage
  - Smart contract compatible
  - Verifiable timestamping

## 📁 Project Structure

```
proof-of-creativity/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── Dockerfile               # Container configuration
├── env.example              # Environment template
├── schema.sql               # Timescale database schema
│
├── app/                     # Main application
│   ├── __init__.py
│   ├── main.py             # FastAPI application
│   ├── config.py           # Configuration
│   │
│   ├── core/               # Core infrastructure
│   │   ├── database.py     # Timescale Vector AI integration
│   │   ├── storage.py      # Multi-backend storage
│   │   └── utils.py        # Utility functions
│   │
│   ├── services/           # Media processing services
│   │   ├── embedding.py    # CLIP embeddings
│   │   ├── fingerprint.py  # Audio fingerprinting
│   │   └── video_processing.py  # Video analysis
│   │
│   ├── models/             # Pydantic data models
│   │   ├── media.py        # Media file models
│   │   └── similarity.py   # Similarity & response models
│   │
│   └── api/                # API routes (future expansion)
│
├── tests/                  # Test suite
├── scripts/                # Utility scripts
├── docs/                   # Documentation
└── venv/                   # Virtual environment
```

## 🛠️ Quick Start

### Prerequisites

- **Python 3.11+**
- **PostgreSQL/Timescale** with pgvector extension
- **FFmpeg** (for video/audio processing)
- **Git** and **pip**

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd proof-of-creativity

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Database Setup

#### Option A: Timescale Cloud (Recommended)
1. Sign up at [Timescale Cloud](https://console.cloud.timescale.com/)
2. Create a new service with Vector AI enabled
3. Copy the connection string

#### Option B: Local PostgreSQL + pgvector
```bash
# Install PostgreSQL and pgvector
# Ubuntu/Debian:
sudo apt install postgresql postgresql-contrib
git clone https://github.com/pgvector/pgvector.git
cd pgvector && make && sudo make install

# Create database
createdb proof_of_creativity
psql proof_of_creativity -c "CREATE EXTENSION vector;"
```

### 3. Configuration

```bash
# Copy environment template
cp env.example .env

# Edit configuration
nano .env
```

**Required Environment Variables:**
```bash
# Database
TIMESCALE_DB_DSN=postgresql://user:password@host:port/database

# Storage (choose one or both)
USE_GCS=true
GCS_BUCKET_NAME=your-bucket-name
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-key.json

USE_WALRUS=false
WALRUS_ENDPOINT=https://api.walrus.xyz

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true
```

### 4. Initialize Database

```bash
# Run database schema
psql $TIMESCALE_DB_DSN -f schema.sql
```

### 5. Run the Application

```bash
# Development mode
python -m app.main

# Or with uvicorn directly
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Test the API

Visit `http://localhost:8000/docs` for interactive API documentation.

**Basic Upload Test:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your-image.jpg"
```

### MySocial Proof of Creativity (oracle)

**Roles:** this service is the **oracle**: it fingerprints media, computes similarity, resolves **`original_creator`** from `media_files.creator_address` (including video frame ids mapped to parent media), maps scores to **integer 0–100** for Move, and submits `analyze_and_update_post` or `analyze_and_update_post_sync_token_pool`. The **chain** applies thresholds and mints/redirects; the **indexer** (separate Postgres) ingests events—keep scores and redirect percentages ≤ 100 so redirection events index cleanly.

### Basic PoC post E2E flow

1. **Post created on-chain** — `post::create_post` with `enable_poc=true` emits `PostCreatedEvent`.
2. **Oracle discovery** — gRPC sync upserts `chain_posts` with `creator_address` (post owner) and enqueues an `analyze_post` job.
3. **Analysis** — `AnalysisService` fingerprints media, queries similarity corpus, resolves `original_creator` from `media_files.creator_address` on matched ids.
4. **Decision** — `DecisionEngine.build_submission` compares score to configured thresholds and decides derivative vs original. **Self-match guard:** when matched `original_creator` equals the posting profile (`chain_posts.creator_address`), the derivative path is skipped (no redirect, no vault provisioning); score is preserved for analytics and a fresh original badge is minted for the new post.
5. **Chain submission** — oracle calls `analyze_and_update_post`; Move clears self-match `original_creator` defensively, then either mints an original badge or applies derivative redirect + optional beneficiary vault.
6. **Indexer** — social indexer writes `poc_analysis_results` (score + score-threshold `similarity_detected`), `poc_badges`, `poc_revenue_redirections`, and vault deposit events into Postgres.
7. **Optional reservation** — `reserve_towards_post_with_platform` tips into the post owner's existing beneficiary vault (not a wrongful derivative vault).

**Self-match rule:** reposts, remasters, and director's cuts by the same creator must not enter the derivative pipeline against themselves. Similarity is still recorded (`highest_similarity_score`); GraphQL `similarityDetected` uses configured thresholds at the indexer layer.

**Local verification:** `ASSUME_YES=1 ./scripts/poc-oracle-post-runnable.sh --self-match-analyze` exercises on-chain self-match semantics (original badge, no `RevenueRedirectionActivatedEvent`).

When `MYSO_INTEGRATION_ENABLED=true`, the service submits Move entries as above when **`MYSO_TOKEN_REGISTRY_ID`** and a per-post **`spt_pool_id`** (query param or inferred from the post object) are available for the sync variant. Required object IDs include **`MYSO_POC_VAULT_DIRECTORY_ID`** in addition to package, config, and registry. If the post has no token pool but you call the sync entry, expect on-chain abort (e.g. `ENoTokenPoolForPost`); the plain `analyze_and_update_post` path is used when pool + token registry are not both configured.

**Chain submission checklist (actually hitting Move):**

1. Set **`MYSO_INTEGRATION_ENABLED=true`** and fill **`MYSO_POC_PACKAGE_ID`**, **`MYSO_POC_CONFIG_ID`**, **`MYSO_POC_REGISTRY_ID`**, **`MYSO_POC_VAULT_DIRECTORY_ID`**, **`MYSOCIAL_RPC_URL`**, and oracle key (**`MYSO_ORACLE_PRIVATE_KEY`** or mnemonic via `myso_wallet`).
2. Ensure the signing address matches **`PoCConfig.oracle_address`** on chain (use **`MYSO_POC_STRICT_ORACLE=true`** to fail client init if it does not).
3. Pass **`post_id`** on **`POST /upload`** or **`POST /upload/stream`**; without it, the API only runs similarity + DB attribution and **does not** submit a PoC transaction.
4. Optionally set **`MYSO_POC_REQUIRE_TX_WHEN_POST_ID=true`** in production so a missing or failed submission returns **502** (sync) or marks the streaming job **failed** instead of completing with `tx_hash=null`.
5. Optional **`MYSO_POC_FAIL_LIFESPAN_ON_MYS_MISCONFIG=true`**: refuse to boot if integration is on but the client cannot initialize (missing IDs, strict oracle, wallet error).
6. **`GET /health`** includes a **`mysocial`** object (`integration_requested`, `client_active`, `oracle_authorized`, `ready_for_submission`). **`GET /readyz`** returns **503** when the database is down or when integration is requested but the oracle is not ready to submit.

- **Prefetch:** uploads with `post_id` call `myso_getObject` and return **409** when the post still has active PoC data (`poc_outcome` ≠ none, populated redirect option, badge snapshot/object). After a dispute **clear**, fields are empty and analysis may run again. Set **`MYSO_POC_ALLOW_FORCE_RESUBMIT=true`** *and* pass `force_reanalyze=true` only for controlled operator reruns (bypasses that preflight).
- **`MYSO_POC_REDIRECT_TARGET`:** `wallet` (0) vs `escrow` (1) for Move `derivative_redirection_target`.
- **`MYSO_POC_CONFIG_CACHE_TTL_SECONDS`:** how long parsed `PoCConfig` stays in memory before refreshing from RPC (limits, thresholds, `max_reasoning_length`, etc.).

**Upload query parameters:** `post_id`, optional `spt_pool_id`, `force_reanalyze`, optional `creator_address` (stored on `media_files` for **`original_creator`** resolution from matched corpus ids), **`royalty_free`** (explicit on-chain outcome 4 when PoC integration is enabled), and `upload_to_stream` on `/upload`.

**Responses:** `UploadResponse` includes **`poc_chain_summary`** (media type code, score %, effective threshold %, whether the derivative-redirect path would apply, explicit-outcome flags). Each **`MediaMatch`** includes optional **`similarity_score_percent`** (0–100, same rounding as Move) alongside **`similarity_score`** (0–1 float). Video responses include **`video_attestation`** (visual vs embedded audio), aligned with Move’s **`embedded_audio_only_derivative`**.

**Persistence:** `attribution_records.proof_data` stores **`poc_submission`** (Move args + `move_function`, `resolved_spt_pool_id`, `derivative_redirection_target`, `tx_hash` when submitted), **`poc_chain_summary`**, **`max_similarity_score_u64`**, **`effective_threshold_u64`**, and **`royalty_free_requested`**. Streaming uploads mirror **`poc_submission`** / **`poc_chain_summary`** on Redis completion for WebSocket/HTTP progress clients.

See `env.example` for variable names and `app/config.py` for defaults.

## 📋 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API information |
| `GET` | `/health` | System health check (includes `mysocial` payload when available) |
| `GET` | `/readyz` | Readiness probe: DB + optional MySocial oracle gate (**503** if not ready) |
| `POST` | `/upload` | Upload and analyze media |
| `GET` | `/media/{id}/matches` | Get similarity matches |
| `GET` | `/stats` | System statistics |

### Example Upload Response

```json
{
  "media_id": "123e4567-e89b-12d3-a456-426614174000",
  "filename": "example.jpg",
  "content_type": "image/jpeg",
  "file_size": 2048576,
  "file_hash": "a1b2c3d4...",
  "storage_uri": "gs://bucket/path/file.jpg",
  "matches": [
    {
      "media_id": "other-media-id",
      "similarity_score": 0.95,
      "match_type": "embedding",
      "confidence_level": "high"
    }
  ],
  "processing_status": "completed",
  "message": "Found 1 high-confidence match - possible derivative content"
}
```

## 🔧 Architecture Deep Dive

### Timescale Vector AI Integration

The system leverages Timescale's Vector AI capabilities for optimal similarity search:

```sql
-- HNSW index for sub-second vector search
CREATE INDEX idx_media_embeddings_vector_hnsw 
ON media_embeddings USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);

-- Time-series + vector filtering
SELECT media_id, similarity_score
FROM media_embeddings
WHERE uploaded_at > NOW() - INTERVAL '24 hours'
  AND (1 - (embedding <=> $1)) >= 0.8
ORDER BY embedding <=> $1
LIMIT 10;
```

### Audio Fingerprinting Algorithm

Implements Shazam-style spectral peak pair hashing:

1. **Spectrogram Generation** - Mel-scaled frequency analysis
2. **Peak Detection** - Local maxima identification
3. **Constellation Mapping** - Time-frequency peak coordinates
4. **Hash Generation** - Frequency pair + time delta hashing
5. **Matching** - Time offset clustering for robustness

**Pipeline:** Steps **1–4** run together in one analysis pass per audio file (`AudioFingerprinter.fingerprint_audio`). Step **5** is separate: it runs at similarity time when `match_fingerprints` compares the query constellation to unpickled `fingerprint_data` from corpus candidates (`verify_constellation_against_blob` → `find_verified_fingerprint_hits`). **Mel fallback:** If `fingerprint_audio_with_blob` catches an error from that primary pipeline, it stores a coarse mel-hash and an **empty** constellation pickle—steps **2–4** are skipped and step **5** verification cannot run for that artifact until a full fingerprint succeeds.

**Query-time behavior:** Steps **1–4** run on every uploaded or embedded audio track. The oracle stores a composite lookup key (`fp_hash`) plus pickled constellation tuples in `audio_fingerprints.fingerprint_data`. Similarity search loads **candidates** that share that `fp_hash`, then applies **step 5** (`match_fingerprints`): agreeing sub-hashes vote on a time offset, and a corpus hit is recorded only after this verification. Reported fingerprint similarity comes from verified confidence (not from “same fp_hash” alone). PostgreSQL holds authoritative rows and blobs; any Redis fingerprint keys are non-authoritative legacy cache.

| Step | Responsibility |
|------|----------------|
| 1–4 (ingest + query audio) | [`app/services/fingerprint.py`](app/services/fingerprint.py) (`AudioFingerprinter`, `fingerprint_audio_with_blob`) |
| Candidate rows | [`app/core/database.py`](app/core/database.py) `search_fingerprint_rows` |
| Step 5 verify | [`app/services/fingerprint.py`](app/services/fingerprint.py) (`verify_constellation_against_blob`, `find_verified_fingerprint_hits`, …) |

#### Inspecting `audio_fingerprints` in PostgreSQL

- **`fp_hash`** — Exactly **40 hexadecimal characters** (SHA-1 digest). This is the stable lookup key you compare across rows or logs.
- **`fingerprint_data`** — **`BYTEA`**: Python **pickle** of constellation tuples. Tools that show BYTEA as hex often display only the first bytes — strings like **`80055D942E`** are typically **pickle protocol framing**, not a second “fingerprint ID”. Ignore short hex previews here; rely on **`fp_hash`** or decode blobs only via [`unpickle_fingerprint_hashes`](app/services/fingerprint.py) in application code.

Sanity check on a row:

```sql
SELECT fp_hash,
       length(fp_hash) AS fp_hash_len,
       octet_length(fingerprint_data) AS blob_octets,
       encode(substring(fingerprint_data from 1 for 8), 'hex') AS blob_prefix_hex
FROM audio_fingerprints
ORDER BY created_at DESC
LIMIT 5;
```

Expect **`fp_hash_len = 40`**. **`blob_prefix_hex`** starting with `8005` / `8004` is normal for pickle; it is not meant to equal `fp_hash`.

### Multi-Modal Processing Pipeline

```mermaid
graph TD
    A[Media Upload] --> B{File Type}
    B -->|Image| C[CLIP Embedding]
    B -->|Audio| D[Spectral Fingerprint]
    B -->|Video| E[Frame + Audio Extract]
    
    C --> F[Vector Search]
    D --> G[Candidate lookup by fp_hash]
    G --> V[Constellation verify + offset clustering]
    V --> I[Similarity Matches]
    E --> H[Batch Process]
    
    F --> I
    H --> I
    
    I --> J[Confidence Scoring]
    J --> K[Attribution Record]
```

## 🧪 Testing

### Run Test Suite

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=app --cov-report=html
```

### Test Individual Components

```bash
# Test database connection
python -c "from app.core.database import check_database_connection; print(check_database_connection())"

# Test embeddings
python -c "from app.services.embedding import image_embedding; print(len(image_embedding('test.jpg')))"

# Test fingerprinting
python -c "from app.services.fingerprint import fingerprint_audio; print(fingerprint_audio('test.wav'))"
```

## 🚀 Deployment

### Docker Deployment

```bash
# Build image
docker build -t proof-of-creativity .

# Run container
docker run -d \
  --name poc-api \
  -p 8000:8000 \
  --env-file .env \
  proof-of-creativity
```

### Railway.com Deployment

1. Connect your GitHub repository
2. Set environment variables in Railway dashboard
3. Deploy automatically on push to main

### Production Considerations

- **Database**: Use Timescale Cloud for optimal vector performance
- **Storage**: Configure appropriate GCS/Walrus buckets
- **Monitoring**: Set up structured logging aggregation
- **Security**: Configure CORS, rate limiting, and authentication
- **Scaling**: Use connection pooling and async processing

## 📊 Performance Benchmarks

### Vector Search Performance (Timescale)
- **1M vectors**: < 10ms average query time
- **10M vectors**: < 50ms average query time
- **HNSW index**: 95%+ recall at 10x speed improvement

### Processing Throughput
- **Images**: ~100/second (CLIP embedding generation)
- **Audio**: ~50/second (fingerprint generation)
- **Video**: ~10/second (frame extraction + processing)

## 🔧 Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TIMESCALE_DB_DSN` | - | Database connection string |
| `USE_GCS` | `true` | Enable Google Cloud Storage |
| `USE_WALRUS` | `false` | Enable Walrus decentralized storage |
| `MAX_FILE_SIZE` | `100MB` | Maximum upload file size |
| `CLIP_MODEL_NAME` | `openai/clip-vit-base-patch32` | CLIP model to use |
| `EMBEDDING_DIMENSION` | `512` | Vector dimension |
| `API_HOST` | `0.0.0.0` | API bind address |
| `API_PORT` | `8000` | API port |
| `DEBUG` | `false` | Enable debug mode |

### Processing Parameters

```python
# Audio fingerprinting
DEFAULT_SAMPLE_RATE = 8000
PEAK_THRESHOLD = 0.1
FAN_VALUE = 5

# Video processing
DEFAULT_KEYFRAME_RATE = 1  # fps
MAX_FRAMES_TO_PROCESS = 300
MAX_VIDEO_DURATION = 600  # seconds

# Similarity thresholds
HIGH_CONFIDENCE = 0.9
MEDIUM_CONFIDENCE = 0.7
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run the test suite: `pytest`
5. Commit your changes: `git commit -am 'Add feature'`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](../../issues)
- **Discussions**: [GitHub Discussions](../../discussions)

## 🔮 Roadmap

- [ ] **Advanced Audio Features**: Genre classification, tempo detection
- [ ] **Video Enhancements**: Scene detection, object recognition
- [ ] **Blockchain Integration**: Smart contract deployment tools
- [ ] **Web Interface**: React-based admin dashboard
- [ ] **Mobile SDK**: React Native/Flutter components
- [ ] **API v2**: GraphQL interface
- [ ] **ML Pipelines**: Custom model training workflows

---

**Built with ❤️ for creators and innovators worldwide.** 