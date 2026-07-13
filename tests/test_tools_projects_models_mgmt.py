"""Project and model management: create, rename, trash, weight downloads."""

from __future__ import annotations

import json

import httpx
import pytest
from fastmcp.exceptions import ToolError

from tests.conftest import SAMPLE_MODEL, client_for

PROJ_ID = "b" * 24
MODEL_ID = "c" * 24


async def test_create_project_derives_slug(app, platform_mock):
    route = platform_mock.post("/api/projects").mock(
        return_value=httpx.Response(201, json={"projectId": PROJ_ID, "slug": "warehouse-safety"})
    )
    async with client_for(app) as client:
        result = await client.call_tool(
            "create_project", {"name": "Warehouse Safety", "visibility": "private"}
        )
    body = json.loads(route.calls[0].request.content)
    assert body == {"name": "Warehouse Safety", "slug": "warehouse-safety", "visibility": "private"}
    assert result.data == {"project_id": PROJ_ID, "slug": "warehouse-safety"}


async def test_create_project_rejects_bad_visibility(app, platform_mock):
    async with client_for(app) as client:
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool("create_project", {"name": "x", "visibility": "secret"})
    assert "visibility" in str(excinfo.value)
    assert not platform_mock.calls


async def test_update_project_by_slug_resolves_then_patches(app, platform_mock):
    platform_mock.get("/api/projects").mock(
        return_value=httpx.Response(
            200,
            json={"projects": [{"_id": PROJ_ID, "name": "old", "slug": "old-name"}], "total": 1},
        )
    )
    route = platform_mock.patch(f"/api/projects/{PROJ_ID}").mock(
        return_value=httpx.Response(200, json={"success": True, "slug": "new-name"})
    )
    async with client_for(app) as client:
        result = await client.call_tool(
            "update_project", {"project": "old-name", "name": "New Name"}
        )
    assert json.loads(route.calls[0].request.content) == {"name": "New Name"}
    assert result.data == {"success": True, "slug": "new-name"}


async def test_update_project_requires_a_field(app, platform_mock):
    async with client_for(app) as client:
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool("update_project", {"project": PROJ_ID})
    assert "at least one" in str(excinfo.value)
    assert not platform_mock.calls


async def test_delete_project_notes_cascade_and_recovery(app, platform_mock):
    route = platform_mock.delete(f"/api/projects/{PROJ_ID}").mock(
        return_value=httpx.Response(200, json={"success": True})
    )
    async with client_for(app) as client:
        result = await client.call_tool("delete_project", {"project": PROJ_ID})
    assert route.called
    note = result.data["note"]
    assert "its models" in note
    assert "restore_from_trash" in note


async def test_get_model_files_maps_signed_urls(app, platform_mock):
    platform_mock.get(f"/api/models/{MODEL_ID}/files").mock(
        return_value=httpx.Response(
            200,
            json={"files": [{"name": "best.pt", "size": 6291456, "downloadUrl": "https://gcs/s"}]},
        )
    )
    async with client_for(app) as client:
        result = await client.call_tool("get_model_files", {"model": MODEL_ID})
    files = result.data["files"]
    assert files == [{"name": "best.pt", "size_bytes": 6291456, "download_url": "https://gcs/s"}]
    assert "short-lived" in result.data["note"]


async def test_get_model_files_empty_says_untrained(app, platform_mock):
    platform_mock.get(f"/api/models/{MODEL_ID}/files").mock(
        return_value=httpx.Response(200, json={"files": []})
    )
    async with client_for(app) as client:
        result = await client.call_tool("get_model_files", {"model": MODEL_ID})
    assert result.data["files"] == []
    assert "complete a training run" in result.data["note"]


async def test_update_model_renames(app, platform_mock):
    route = platform_mock.patch(f"/api/models/{MODEL_ID}").mock(
        return_value=httpx.Response(200, json={"success": True, "slug": "exp-renamed"})
    )
    async with client_for(app) as client:
        result = await client.call_tool("update_model", {"model": MODEL_ID, "name": "Exp Renamed"})
    assert json.loads(route.calls[0].request.content) == {"name": "Exp Renamed"}
    assert result.data == {"success": True, "slug": "exp-renamed"}


async def test_delete_model_soft_deletes(app, platform_mock):
    route = platform_mock.delete(f"/api/models/{MODEL_ID}").mock(
        return_value=httpx.Response(200, json={"success": True})
    )
    async with client_for(app) as client:
        result = await client.call_tool("delete_model", {"model": MODEL_ID})
    assert route.called
    assert "restore_from_trash" in result.data["note"]


async def test_model_slug_ambiguity_returns_candidates(app, platform_mock):
    twins = [
        {**SAMPLE_MODEL, "_id": "m1" + "0" * 22},
        {**SAMPLE_MODEL, "_id": "m2" + "0" * 22},
    ]
    platform_mock.get("/api/models").mock(return_value=httpx.Response(200, json={"models": twins}))
    async with client_for(app) as client:
        result = await client.call_tool("update_model", {"model": "detector-v1", "name": "x"})
    data = result.data
    assert "success" not in data
    assert len(data["candidates"]) == 2  # never a silent pick
