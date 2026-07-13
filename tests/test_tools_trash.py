"""Trash: list with expiry countdown, restore, confirm-gated permanent purge."""

from __future__ import annotations

import json

import httpx
import pytest
from fastmcp.exceptions import ToolError

from tests.conftest import client_for

TRASH_ITEM = {
    "_id": "f" * 24,
    "type": "dataset",
    "name": "old-forklifts",
    "slug": "old-forklifts",
    "trashedAt": "2026-07-01T00:00:00Z",
    "daysRemaining": 18,
    "sizeBytes": 104857600,
}


async def test_list_trash_maps_expiry_and_summary(app, platform_mock):
    platform_mock.get("/api/trash").mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [TRASH_ITEM],
                "total": 1,
                "summary": {"totalItems": 1, "totalSizeBytes": 104857600},
            },
        )
    )
    async with client_for(app) as client:
        result = await client.call_tool("list_trash", {})
    data = result.data
    assert data["items"][0]["days_remaining"] == 18
    assert data["items"][0]["size_bytes"] == 104857600
    assert data["total_size_bytes"] == 104857600


async def test_list_trash_type_filter_and_empty_note(app, platform_mock):
    route = platform_mock.get("/api/trash", params={"type": "model"}).mock(
        return_value=httpx.Response(200, json={"items": [], "total": 0, "summary": {}})
    )
    async with client_for(app) as client:
        result = await client.call_tool("list_trash", {"type": "model"})
    assert route.called
    assert result.data["note"] == "Trash is empty."


async def test_list_trash_rejects_bad_type(app, platform_mock):
    async with client_for(app) as client:
        with pytest.raises(ToolError):
            await client.call_tool("list_trash", {"type": "export"})
    assert not platform_mock.calls


async def test_restore_project_reports_models(app, platform_mock):
    route = platform_mock.post("/api/trash").mock(
        return_value=httpx.Response(200, json={"success": True, "restoredModels": 3})
    )
    async with client_for(app) as client:
        result = await client.call_tool(
            "restore_from_trash", {"item_id": "b" * 24, "type": "project"}
        )
    assert json.loads(route.calls[0].request.content) == {"id": "b" * 24, "type": "project"}
    data = result.data
    assert data["success"] is True
    assert data["restored_models"] == 3
    assert "3 of its models" in data["note"]


async def test_purge_requires_confirm(app, platform_mock):
    async with client_for(app) as client:
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool("purge_from_trash", {"item_id": "f" * 24, "type": "dataset"})
    message = str(excinfo.value)
    assert "permanent" in message
    assert "confirm=true" in message
    assert not platform_mock.calls


async def test_purge_with_confirm_deletes(app, platform_mock):
    route = platform_mock.delete("/api/trash").mock(
        return_value=httpx.Response(200, json={"success": True, "deletedCount": 1})
    )
    async with client_for(app) as client:
        result = await client.call_tool(
            "purge_from_trash", {"item_id": "f" * 24, "type": "dataset", "confirm": True}
        )
    assert json.loads(route.calls[0].request.content) == {"id": "f" * 24, "type": "dataset"}
    assert result.data["deleted_count"] == 1
    assert "storage freed" in result.data["note"]
