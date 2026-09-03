"""Convert official SDK failures into safe, actionable MCP tool errors."""

from __future__ import annotations

import functools
import re
from collections.abc import Callable
from typing import Any

from fastmcp.exceptions import ToolError
from ultralytics_platform import APIConnectionError, APIError


def _reason(error: APIError) -> str | None:
    detail = error.json
    if not isinstance(detail, dict):
        return None
    value = detail.get("error") or detail.get("message")
    if not isinstance(value, str) or not value.strip():
        return None
    return re.sub(r"ul_[A-Za-z0-9_-]+", "[REDACTED]", value.strip())[:300]


def translate(error: APIError | APIConnectionError) -> ToolError:
    if isinstance(error, APIConnectionError):
        return ToolError(
            "The connection to the Ultralytics Platform failed. Check the resource's "
            "current status before retrying a mutation because the request outcome is unknown."
        )

    status = error.status_code
    reason = _reason(error)
    request = f" Request ID: {error.request_id}." if error.request_id else ""
    if status == 401:
        message = (
            "ULTRALYTICS_API_KEY is missing or invalid. Create or replace it at "
            "https://platform.ultralytics.com/settings?tab=api-keys."
        )
    elif status == 402:
        message = reason or "Your account does not have enough credits for this operation."
        message += " Add credits in Platform billing and try again."
    elif status == 403:
        message = reason or "Your account cannot perform this operation."
        message += " Check workspace permissions, plan limits, and quotas."
    elif status == 404:
        message = reason or "The requested Platform resource was not found."
        message += " Check the owner and resource slugs."
    elif status == 409:
        message = reason or "A conflicting operation is already in progress."
        message += " Inspect the current resource status instead of retrying immediately."
    elif status == 429:
        message = "The Platform is rate-limiting requests. Wait briefly and try again."
    elif status >= 500:
        message = "The Ultralytics Platform is temporarily unavailable. Try again later."
    else:
        message = reason or f"The Platform rejected the request with HTTP {status}."
    return ToolError(message + request)


def sdk_errors(function: Callable[..., Any]) -> Callable[..., Any]:
    """Apply uniform SDK error handling to an asynchronous MCP tool."""

    @functools.wraps(function)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            return await function(*args, **kwargs)
        except (APIError, APIConnectionError) as error:
            raise translate(error) from None

    return wrapped
