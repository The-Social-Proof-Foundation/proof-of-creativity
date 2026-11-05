"""Shared gRPC transport helpers."""

from __future__ import annotations

from urllib.parse import urlparse


def parse_grpc_target(grpc_url: str, grpc_tls: bool) -> tuple[str, bool]:
    """Return (host:port, use_tls) from profile grpc_url."""
    if "://" not in grpc_url:
        return grpc_url, grpc_tls
    parsed = urlparse(grpc_url)
    host = parsed.hostname or grpc_url
    if parsed.port:
        port = parsed.port
    elif parsed.scheme == "https":
        port = 443
    else:
        port = 9000
    use_tls = grpc_tls or parsed.scheme == "https"
    return f"{host}:{port}", use_tls
