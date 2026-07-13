"""Account insight and trash tools (US4, FR-005)."""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

from fastmcp.exceptions import ToolError
from pydantic import Field

from ..auth import get_request_token
from ..errors import platform_errors
from ..platform_client import platform_api
from ..schemas import (
    AccountStatus,
    ActivityItem,
    TrashItem,
    bounded_dump,
    clamp_limit,
    make_list_result,
)

TRASH_TYPES = ("project", "dataset", "model")


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


@platform_errors
async def list_trash(
    type: Annotated[
        str | None, Field(description="Filter: 'project', 'dataset' or 'model' (default: all)")
    ] = None,
    limit: Annotated[
        int | None, Field(description="Max items to return (default 20, max 50)", ge=1)
    ] = None,
    page: Annotated[int | None, Field(description="Page number for older items", ge=1)] = None,
) -> dict[str, Any]:
    """See what's in the trash and how long until it's gone for good.

    Read-only — spends nothing. Each item shows days_remaining before the daily
    cleanup deletes it permanently (30-day retention). Trashed items still count
    toward storage — purge_from_trash frees the space immediately.
    """
    if type is not None and type not in TRASH_TYPES:
        raise ToolError(f"type must be one of: {', '.join(TRASH_TYPES)} (or omitted for all).")
    token = get_request_token()
    params: dict[str, Any] = {"limit": clamp_limit(limit)}
    if type:
        params["type"] = type
    if page is not None:
        params["page"] = page
    data = await platform_api.get(
        "/api/trash", token=token, params=params, resource_hint="Your trash"
    )
    items = [TrashItem.from_api(item) for item in data.get("items", [])]
    result = make_list_result(
        items,
        total=data.get("total"),
        empty_note="Trash is empty.",
    )
    summary = data.get("summary") or {}
    if items and summary.get("totalSizeBytes") is not None:
        result["total_size_bytes"] = summary["totalSizeBytes"]
    return result


@platform_errors
async def restore_from_trash(
    item_id: Annotated[str, Field(description="The trashed item's id (from list_trash)")],
    type: Annotated[str, Field(description="What it is: 'project', 'dataset' or 'model'")],
) -> dict[str, Any]:
    """Restore a trashed project, dataset or model.

    State-changing — spends no credits. The item returns to your account exactly
    where it was; restoring a project brings its models back with it.
    """
    if type not in TRASH_TYPES:
        raise ToolError(f"type must be one of: {', '.join(TRASH_TYPES)}.")
    token = get_request_token()
    data = await platform_api.post(
        "/api/trash",
        token=token,
        json={"id": item_id, "type": type},
        resource_hint=f"Trashed {type} '{item_id}'",
    )
    result: dict[str, Any] = {"success": bool(data.get("success")), "restored": type}
    if data.get("restoredModels"):
        result["restored_models"] = data["restoredModels"]
        result["note"] = f"Project restored along with {data['restoredModels']} of its models."
    return result


@platform_errors
async def purge_from_trash(
    item_id: Annotated[str, Field(description="The trashed item's id (from list_trash)")],
    type: Annotated[str, Field(description="What it is: 'project', 'dataset' or 'model'")],
    confirm: Annotated[
        bool,
        Field(
            description="Must be true — purging is PERMANENT and cascades (a project "
            "takes its models, exports and deployments with it)"
        ),
    ] = False,
) -> dict[str, Any]:
    """Permanently delete a trashed item now, freeing its storage.

    State-changing and IRREVERSIBLE — the item and everything cascading from it
    (models, exports, deployments, stored files) are removed immediately instead of
    waiting out the 30-day window. Spends no credits.
    """
    if type not in TRASH_TYPES:
        raise ToolError(f"type must be one of: {', '.join(TRASH_TYPES)}.")
    if not confirm:
        raise ToolError(
            f"Purging this {type} is permanent and cannot be undone (cascades to "
            "models/exports/deployments and stored files). Call again with "
            "confirm=true to proceed — or leave it to expire on its own."
        )
    token = get_request_token()
    data = await platform_api.delete(
        "/api/trash",
        token=token,
        json={"id": item_id, "type": type},
        resource_hint=f"Trashed {type} '{item_id}'",
    )
    result: dict[str, Any] = {
        "success": bool(data.get("success")),
        "deleted_count": data.get("deletedCount"),
        "note": "Permanently deleted — storage freed.",
    }
    if data.get("cascadedModels"):
        result["cascaded_models"] = data["cascadedModels"]
    return result


def register(mcp) -> None:
    mcp.tool(get_account_status)
    mcp.tool(get_recent_activity)
    mcp.tool(list_trash)
    mcp.tool(restore_from_trash)
    mcp.tool(purge_from_trash)
