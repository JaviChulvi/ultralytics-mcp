"""Per-request credential extraction (FR-002).

The credential arrives on every MCP request as an HTTP Authorization header, set once
in the user's assistant configuration. It is read here, handed to the platform client,
and never stored or logged.
"""

from __future__ import annotations

from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_http_headers

from .errors import CREDENTIAL_GUIDANCE


def get_request_token() -> str:
    """Return the Bearer token from the current request, or raise credential guidance."""
    headers = get_http_headers(include={"authorization"})
    authorization = headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token:
        raise ToolError(CREDENTIAL_GUIDANCE)
    return token
