from __future__ import annotations

import ipaddress


def is_loopback_address(host: str) -> bool:
    """Return True for loopback peer addresses accepted by the RPC server."""

    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host in {"localhost"}


def bearer_token(authorization: str | None) -> str | None:
    """Extract a bearer token without logging or otherwise exposing it."""

    if not authorization:
        return None
    scheme, separator, token = authorization.partition(" ")
    if separator and scheme.lower() == "bearer" and token:
        return token.strip() or None
    return None
