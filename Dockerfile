# Multi-stage build for PoC oracle (Python 3.10 — matches production/Railway)
FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-blockchain.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt \
    && pip install --no-cache-dir --user -r requirements-blockchain.txt

FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY --from=builder /root/.local /root/.local

WORKDIR /app
COPY . .

RUN mkdir -p /app/uploads/image /app/uploads/audio /app/uploads/video \
    && mkdir -p /tmp/proof-of-creativity \
    && mkdir -p /app/data/blobs

ENV PATH=/root/.local/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '8000') + '/health', timeout=5)" || exit 1

CMD ["python", "scripts/run_api.py"]
