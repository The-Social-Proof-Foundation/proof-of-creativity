import os

# Storage backend toggles
USE_CLOUDFLARE_R2 = os.getenv("USE_CLOUDFLARE_R2", "true").lower() == "true"
USE_GCS = os.getenv("USE_GCS", "false").lower() == "true"
USE_WALRUS = os.getenv("USE_WALRUS", "false").lower() == "true"

# Vector DB Settings
VECTOR_DB = "postgresql"

# Cloudflare R2 (Primary storage for CDN delivery - all media types)
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "dripdrop-media")
R2_PUBLIC_DOMAIN = os.getenv("R2_PUBLIC_DOMAIN", "")  # e.g., cdn.dripdrop.app

# Cloudflare Stream (Video streaming platform - opt-in per upload)
# Stream is optimized for video delivery with automatic encoding and adaptive bitrate
# Usage: Set upload_to_stream=true in API call to enable Stream upload for that video
STREAM_ACCOUNT_ID = os.getenv("STREAM_ACCOUNT_ID", "")
STREAM_API_TOKEN = os.getenv("STREAM_API_TOKEN", "")
STREAM_CUSTOMER_SUBDOMAIN = os.getenv("STREAM_CUSTOMER_SUBDOMAIN", "")  # Optional custom domain

# GCS (Optional backup storage)
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "proof-of-creativity-testnet")
GCS_BACKUP_ENABLED = os.getenv("GCS_BACKUP_ENABLED", "false").lower() == "true"

# Walrus (Decentralized storage)
WALRUS_ENDPOINT = os.getenv("WALRUS_ENDPOINT", "https://api.walrus.xyz")

# Rate Limiting (Redis-based)
RATE_LIMIT_UPLOADS_PER_HOUR = int(os.getenv("RATE_LIMIT_UPLOADS_PER_HOUR", "100"))

# MySocial Blockchain Integration (optional)
MYSO_INTEGRATION_ENABLED = os.getenv("MYSO_INTEGRATION_ENABLED", "false").lower() == "true"
MYSOCIAL_RPC_URL = os.getenv("MYSOCIAL_RPC_URL", "https://fullnode.testnet.mysocial:9000")
MYSO_POC_PACKAGE_ID = os.getenv("MYSO_POC_PACKAGE_ID", "")
MYSO_POC_CONFIG_ID = os.getenv("MYSO_POC_CONFIG_ID", "")
MYSO_POC_REGISTRY_ID = os.getenv("MYSO_POC_REGISTRY_ID", "")
MYSO_POC_VAULT_DIRECTORY_ID = os.getenv("MYSO_POC_VAULT_DIRECTORY_ID", "")
MYSO_TOKEN_REGISTRY_ID = os.getenv("MYSO_TOKEN_REGISTRY_ID", "")
# wallet | escrow → integer redirect target consumed by Move (see app.services.poc_utils)
MYSO_POC_REDIRECT_TARGET = os.getenv("MYSO_POC_REDIRECT_TARGET", "wallet")
MYSO_POC_CONFIG_CACHE_TTL_SECONDS = os.getenv("MYSO_POC_CONFIG_CACHE_TTL_SECONDS", "60")
MYSO_POC_GAS_BUDGET = os.getenv("MYSO_POC_GAS_BUDGET", "20000000")
MYSO_POC_GAS_PRICE = os.getenv("MYSO_POC_GAS_PRICE", "1000")
