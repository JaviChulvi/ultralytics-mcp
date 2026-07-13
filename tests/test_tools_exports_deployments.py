"""Export jobs and deployment lifecycle: create, poll, remove."""

from __future__ import annotations

import json

import httpx
import pytest
from fastmcp.exceptions import ToolError

from tests.conftest import SAMPLE_DEPLOYMENT, client_for

MODEL_ID = "c" * 24
DEP_ID = "d" * 24
EXPORT_ID = "e" * 24


async def test_create_export_builds_args_and_queues(app, platform_mock):
    route = platform_mock.post("/api/exports").mock(
        return_value=httpx.Response(
            201, json={"exportId": EXPORT_ID, "format": "onnx", "status": "queued"}
        )
    )
    async with client_for(app) as client:
        result = await client.call_tool(
            "create_export",
            {"model": MODEL_ID, "format": "onnx", "imgsz": 640, "simplify": True},
        )
    assert json.loads(route.calls[0].request.content) == {
        "modelId": MODEL_ID,
        "format": "onnx",
        "args": {"imgsz": 640, "simplify": True},
    }
    data = result.data
    assert data["export_id"] == EXPORT_ID
    assert "poll get_export" in data["note"]


@pytest.mark.parametrize(
    ("args", "fragment"),
    [
        ({"model": MODEL_ID, "format": "tensorflowjs"}, "format"),
        ({"model": MODEL_ID, "format": "engine"}, "gpu_type"),
    ],
)
async def test_create_export_validates_before_calling(app, platform_mock, args, fragment):
    async with client_for(app) as client:
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool("create_export", args)
    assert fragment in str(excinfo.value)
    assert not platform_mock.calls


async def test_create_export_engine_sends_gpu(app, platform_mock):
    route = platform_mock.post("/api/exports").mock(
        return_value=httpx.Response(
            201, json={"exportId": EXPORT_ID, "format": "engine", "status": "starting"}
        )
    )
    async with client_for(app) as client:
        await client.call_tool(
            "create_export", {"model": MODEL_ID, "format": "engine", "gpu_type": "rtx-4090"}
        )
    assert json.loads(route.calls[0].request.content) == {
        "modelId": MODEL_ID,
        "format": "engine",
        "gpuType": "rtx-4090",
    }


async def test_create_export_duplicate_conflict_names_export(app, platform_mock):
    platform_mock.post("/api/exports").mock(
        return_value=httpx.Response(
            409, json={"error": "An onnx export is already in progress.", "exportId": "exp_dup"}
        )
    )
    async with client_for(app) as client:
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool("create_export", {"model": MODEL_ID, "format": "onnx"})
    message = str(excinfo.value)
    assert "already in progress" in message
    assert "exp_dup" in message


async def test_get_export_completed_maps_file_object(app, platform_mock):
    platform_mock.get(f"/api/exports/{EXPORT_ID}").mock(
        return_value=httpx.Response(
            200,
            json={
                "export": {
                    "_id": EXPORT_ID,
                    "modelId": MODEL_ID,
                    "format": "onnx",
                    "status": "completed",
                    "file": {
                        "path": "exports/model.onnx",
                        "size": 12582912,
                        "downloadUrl": "https://gcs/signed",
                        "downloadFilename": "detector-v1.onnx",
                    },
                    "completedAt": "2026-07-13T10:00:00Z",
                }
            },
        )
    )
    async with client_for(app) as client:
        result = await client.call_tool("get_export", {"export_id": EXPORT_ID})
    data = result.data
    assert data["file"] == "detector-v1.onnx"
    assert data["file_size_bytes"] == 12582912
    assert data["download_url"] == "https://gcs/signed"
    assert "signed" in data["note"]


async def test_get_export_running_says_poll_again(app, platform_mock):
    platform_mock.get(f"/api/exports/{EXPORT_ID}").mock(
        return_value=httpx.Response(
            200, json={"export": {"_id": EXPORT_ID, "format": "engine", "status": "running"}}
        )
    )
    async with client_for(app) as client:
        result = await client.call_tool("get_export", {"export_id": EXPORT_ID})
    assert "poll again" in result.data["note"].lower()


@pytest.mark.parametrize(
    ("action", "fragment"),
    [("cancelled", "cancelled"), ("deleted", "artifact deleted")],
)
async def test_delete_export_reports_action(app, platform_mock, action, fragment):
    platform_mock.delete(f"/api/exports/{EXPORT_ID}").mock(
        return_value=httpx.Response(200, json={"success": True, "action": action})
    )
    async with client_for(app) as client:
        result = await client.call_tool("delete_export", {"export_id": EXPORT_ID})
    assert result.data["action"] == action
    assert fragment in result.data["note"]


async def test_create_deployment_queues_provisioning(app, platform_mock):
    route = platform_mock.post("/api/deployments").mock(
        return_value=httpx.Response(
            201,
            json={
                "deploymentId": DEP_ID,
                "status": "deploying",
                "message": "Deployment started.",
                "region": "europe-southwest1",
            },
        )
    )
    async with client_for(app) as client:
        result = await client.call_tool(
            "create_deployment",
            {"model": MODEL_ID, "name": "detector-prod", "region": "europe-southwest1"},
        )
    assert json.loads(route.calls[0].request.content) == {
        "modelId": MODEL_ID,
        "name": "detector-prod",
        "region": "europe-southwest1",
    }
    data = result.data
    assert data["deployment_id"] == DEP_ID
    assert data["status"] == "deploying"
    assert "poll get_deployment" in data["note"]


async def test_create_deployment_quota_error_is_actionable(app, platform_mock):
    platform_mock.post("/api/deployments").mock(
        return_value=httpx.Response(
            403,
            json={
                "error": "Deployment quota exceeded",
                "quotaType": "deployments",
                "current": 3,
                "limit": 3,
            },
        )
    )
    async with client_for(app) as client:
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool(
                "create_deployment",
                {"model": MODEL_ID, "name": "x", "region": "us-central1"},
            )
    message = str(excinfo.value)
    assert "deployments quota" in message
    assert "3/3" in message


async def test_delete_deployment_requires_confirm(app, platform_mock):
    async with client_for(app) as client:
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool("delete_deployment", {"deployment": DEP_ID})
    assert "confirm=true" in str(excinfo.value)
    assert not platform_mock.calls


async def test_delete_deployment_by_name_with_confirm(app, platform_mock):
    platform_mock.get("/api/deployments").mock(
        return_value=httpx.Response(
            200, json={"deployments": [{**SAMPLE_DEPLOYMENT, "_id": DEP_ID}], "total": 1}
        )
    )
    route = platform_mock.delete(f"/api/deployments/{DEP_ID}").mock(
        return_value=httpx.Response(200, json={"success": True, "message": "Deployment deleted"})
    )
    async with client_for(app) as client:
        result = await client.call_tool(
            "delete_deployment", {"deployment": "prod-endpoint", "confirm": True}
        )
    assert route.called
    assert result.data["success"] is True
    assert "permanently" in result.data["note"]
