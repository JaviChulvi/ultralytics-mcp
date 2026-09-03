"""Small shared helpers for SDK keyword arguments."""

from __future__ import annotations

from typing import Any


def provided(**values: Any) -> dict[str, Any]:
    """Return only explicitly provided values so SDK defaults remain authoritative."""
    return {key: value for key, value in values.items() if value is not None}


def query_bool(value: bool | None) -> str | None:
    if value is None:
        return None
    return "true" if value else "false"
