#!/usr/bin/env bash
# Generate Python gRPC stubs for MySocial gRPC APIs (v2 + alpha).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROTO_DIR="$ROOT/proto/defs"
OUT_DIR="$ROOT/app/chain/proto"

mkdir -p "$OUT_DIR"

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON="$(cd "$(dirname "$0")/.." && pwd)/.venv/bin/python"
fi

PROTO_FILES=()
while IFS= read -r proto_file; do
  PROTO_FILES+=("$proto_file")
done < <(find "$PROTO_DIR" -name "*.proto" | sort)

"$PYTHON" -m grpc_tools.protoc \
  -I"$PROTO_DIR" \
  --python_out="$OUT_DIR" \
  --grpc_python_out="$OUT_DIR" \
  "${PROTO_FILES[@]}"

# grpc_tools emits myso/google.rpc imports without package prefix; fix those only.
ROOT="$ROOT" "$PYTHON" - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["ROOT"]) / "app" / "chain" / "proto"
replacements = [
    ("from google.rpc import", "from app.chain.proto.google.rpc import"),
    ("from myso.rpc.alpha import", "from app.chain.proto.myso.rpc.alpha import"),
    ("from myso.rpc.v2 import", "from app.chain.proto.myso.rpc.v2 import"),
]

for path in root.rglob("*_pb2*.py"):
    text = path.read_text(encoding="utf-8")
    original = text
    for old, new in replacements:
        text = text.replace(old, new)
    if text != original:
        path.write_text(text, encoding="utf-8")
PY

echo "Generated gRPC stubs in $OUT_DIR"
