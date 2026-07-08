"""Translate upstream platform failures into actionable user-facing messages.

Every failure names its cause and a next step (FR-007, SC-004). Nothing internal —
status codes aside, no tracebacks, and never the credential — reaches the user.
"""

from __future__ import annotations

import functools

from fastmcp.exceptions import ToolError

CREDENTIAL_GUIDANCE = (
    "Your Ultralytics Platform API key is missing or was rejected. Create one at "
    "https://platform.ultralytics.com under Settings > API Keys, add it to your "
    "assistant's MCP configuration as an 'Authorization: Bearer ul_...' header, "
    "then try again."
)


class PlatformError(Exception):
    """An upstream platform call failed.

    ``status`` is the HTTP status code, or ``None`` for network/timeout failures.
    ``resource_hint`` names what was being looked up, for useful 404 messages.
    """

    def __init__(self, status: int | None, resource_hint: str | None = None):
        self.status = status
        self.resource_hint = resource_hint
        super().__init__(f"platform request failed (status={status})")


def translate(error: PlatformError) -> ToolError:
    status = error.status
    hint = error.resource_hint or "The requested resource"
    if status == 401:
        message = CREDENTIAL_GUIDANCE
    elif status == 403:
        message = (
            f"{hint} exists but your account doesn't have access to it. "
            "Check that you're using the right account's API key."
        )
    elif status == 404:
        message = (
            f"{hint} was not found on the platform. Check the id or name — "
            "the list tools can help find the right one."
        )
    elif status == 429:
        message = (
            "The platform is rate-limiting requests right now. "
            "Wait a moment and try again — retrying is safe."
        )
    elif status is not None and status >= 500:
        message = (
            "The Ultralytics Platform is temporarily unavailable (server error). "
            "Nothing was changed; retrying later is safe."
        )
    elif status is None:
        message = (
            "Could not reach the Ultralytics Platform (network problem or timeout). "
            "Nothing was changed; retrying later is safe."
        )
    else:
        message = (
            f"The platform rejected the request (HTTP {status}). "
            "Check the tool inputs and try again."
        )
    return ToolError(message)


def platform_errors(fn):
    """Uniform failure behavior for every tool: PlatformError → actionable ToolError (D7)."""

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except PlatformError as exc:
            raise translate(exc) from None

    return wrapper
