"""Account insight tools (US4, FR-005). All read-only."""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

from pydantic import Field

from ..auth import get_request_token
from ..errors import platform_errors
from ..platform_client import platform_api
from ..schemas import AccountStatus, ActivityItem, bounded_dump, clamp_limit, make_list_result


@platform_errors
async def get_account_status() -> dict[str, Any]:
    """Check your credit balance, plan and storage usage in one call.

    Read-only — spends nothing. Returns remaining credits (in cents), your plan,
    storage tier and bytes used.
    """
    token = get_request_token()
    balance, storage = await asyncio.gather(
        platform_api.get("/api/billing/balance", token=token, resource_hint="Your balance"),
        platform_api.get("/api/storage", token=token, resource_hint="Your storage usage"),
    )
    usage = storage.get("usage") or {}
    status = AccountStatus(
        credits_cents=balance.get("creditsCents"),
        plan=balance.get("plan"),
        storage_tier=storage.get("tier"),
        storage_used_bytes=usage.get("storage"),
    )
    return bounded_dump(status)


@platform_errors
async def get_recent_activity(
    limit: Annotated[
        int | None, Field(description="Max events to return (default 20, max 50)", ge=1)
    ] = None,
    page: Annotated[int | None, Field(description="Page number for older events", ge=1)] = None,
) -> dict[str, Any]:
    """See what happened on your account recently (a summarized activity feed).

    Read-only — spends nothing. Returns recent events — action, resource type and
    name, timestamp — newest first, with page-based access to older events.
    """
    token = get_request_token()
    params: dict[str, Any] = {"limit": clamp_limit(limit)}
    if page is not None:
        params["page"] = page
    data = await platform_api.get(
        "/api/activity", token=token, params=params, resource_hint="Your activity feed"
    )
    events = [ActivityItem.from_api(item) for item in data.get("events", [])]
    return make_list_result(
        events,
        total=data.get("total"),
        empty_note="No recent activity on this account.",
    )


def register(mcp) -> None:
    mcp.tool(get_account_status)
    mcp.tool(get_recent_activity)
