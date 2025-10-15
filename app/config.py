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

# GCS (Optional backup storage)
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "proof-of-creativity-testnet")
GCS_BACKUP_ENABLED = os.getenv("GCS_BACKUP_ENABLED", "false").lower() == "true"

# Walrus (Decentralized storage)
WALRUS_ENDPOINT = os.getenv("WALRUS_ENDPOINT", "https://api.walrus.xyz")
