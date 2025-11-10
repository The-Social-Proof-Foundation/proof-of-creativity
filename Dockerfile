# Multi-stage build for optimized production image
FROM python:3.10-slim as builder

# Install system dependencies for building packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.10-slim

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Create app directory and copy application code
WORKDIR /app
COPY . .

# Create necessary directories
RUN mkdir -p /app/uploads/image /app/uploads/audio /app/uploads/video \
    && mkdir -p /tmp/proof-of-creativity

# Ensure the PATH includes user site-packages
ENV PATH=/root/.local/bin:$PATH

# Expose port (Railway sets PORT env var)
EXPOSE 8080

# Health check using Python
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:' + os.getenv('PORT', '8080') + '/health')" || exit 1

# Start using Python directly (reads PORT from environment in app/main.py)
CMD ["python", "-m", "app.main"]
