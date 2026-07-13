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
    ``detail`` is the parsed error body when the platform sent one — its ``error``
    text, quota numbers and in-flight job ids make the translated message concrete.
    """

    def __init__(
        self,
        status: int | None,
        resource_hint: str | None = None,
        detail: dict | None = None,
    ):
        self.status = status
        self.resource_hint = resource_hint
        self.detail = detail if isinstance(detail, dict) else None
        super().__init__(f"platform request failed (status={status})")


def _upstream_reason(detail: dict | None) -> str | None:
    """The platform's own error text, bounded — platform-authored, never internal."""
    if not detail:
        return None
    reason = detail.get("error") or detail.get("message")
    if isinstance(reason, str) and reason.strip():
        return reason.strip()[:300]
    return None


def translate(error: PlatformError) -> ToolError:
    status = error.status
    hint = error.resource_hint or "The requested resource"
    detail = error.detail or {}
    reason = _upstream_reason(error.detail)
    if status == 401:
        message = CREDENTIAL_GUIDANCE
    elif status == 402:
        message = (
            f"{reason or 'Your account balance is too low for this operation.'} "
            "Top up credits at platform.ultralytics.com under Settings > Billing, "
            "then try again."
        )
    elif status == 403:
        if detail.get("quotaType"):
            current, limit = detail.get("current"), detail.get("limit")
            usage = f" ({current}/{limit} used)" if current is not None else ""
            message = (
                f"Your plan's {detail['quotaType']} quota is full{usage}. "
                "Free up capacity or upgrade the plan at platform.ultralytics.com."
            )
        elif reason:
            message = f"{reason} Upgrade options are at platform.ultralytics.com."
        else:
            message = (
                f"{hint} exists but your account doesn't have access to it. "
                "Check that you're using the right account's API key."
            )
    elif status == 404:
        message = (
            f"{hint} was not found on the platform. Check the id or name — "
            "the list tools can help find the right one."
        )
    elif status == 409:
        in_flight = detail.get("existingJobId") or detail.get("exportId")
        job_note = f" (in-flight id: {in_flight})" if in_flight else ""
        conflict = reason or f"{hint} already has this operation in progress."
        message = f"{conflict}{job_note} Wait for the current operation to finish, then retry."
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
    elif reason:
        message = f"The platform rejected the request (HTTP {status}): {reason}"
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
