"""GPU availability and the confirmation-gated training start."""

from __future__ import annotations

import json

import httpx
import pytest
from fastmcp.exceptions import ToolError

from tests.conftest import SAMPLE_DATASET, SAMPLE_PROJECT, client_for

DS_ID = "a" * 24
PROJ_ID = "b" * 24
MODEL_ID = "c" * 24

START_RESPONSE = {
    "modelId": MODEL_ID,
    "instanceId": "inst_1",
    "status": "starting",
    "gpuType": "rtx-4090",
    "estimatedCost": {"pricePerHour": 0.69, "gpuMemoryGb": 24},
    "billing": {
        "estimatedCostCents": 173,
        "estimatedCostDisplay": "$1.73",
        "balanceCents": 2500,
    },
}


def _mock_happy_path(platform_mock):
    platform_mock.get(f"/api/datasets/{DS_ID}").mock(
        return_value=httpx.Response(
            200,
            json={
                "dataset": {
                    **SAMPLE_DATASET,
                    "_id": DS_ID,
                    "username": "javier",
                    "slug": "forklifts",
                }
            },
        )
    )
    platform_mock.get("/api/projects").mock(
        return_value=httpx.Response(
            200, json={"projects": [{**SAMPLE_PROJECT, "_id": PROJ_ID}], "total": 1}
        )
    )
    shell = platform_mock.post("/api/models").mock(
        return_value=httpx.Response(201, json={"modelId": MODEL_ID, "slug": "exp"})
    )
    start = platform_mock.post("/api/training/start").mock(
        return_value=httpx.Response(200, json=START_RESPONSE)
    )
    return shell, start


async def test_gpu_availability_merges_pricing_and_stock(app, platform_mock):
    platform_mock.get("/api/training/gpu-availability").mock(
        return_value=httpx.Response(200, json={"rtx-4090": "High", "b200": "Low"})
    )
    async with client_for(app) as client:
        result = await client.call_tool("get_gpu_availability", {})
    gpus = {g["id"]: g for g in result.data["gpus"]}
    assert gpus["rtx-4090"]["availability"] == "High"
    assert gpus["rtx-4090"]["price_per_hour_usd"] == 0.69
    assert gpus["b200"]["min_tier"] == "pro"
    assert gpus["h100-sxm"]["availability"] is None


async def test_start_training_refuses_without_confirmation(app, platform_mock):
    async with client_for(app) as client:
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool(
                "start_training", {"dataset": DS_ID, "project": "warehouse-safety"}
            )
    message = str(excinfo.value)
    assert "SPENDS CREDITS" in message
    assert "confirm_spend=true" in message
    assert "$0.69" in message  # names the default GPU's rate
    assert "web UI" in message  # the cancel gap is disclosed up front
    assert not platform_mock.calls


async def test_start_training_choreography_and_response(app, platform_mock):
    shell, start = _mock_happy_path(platform_mock)
    async with client_for(app) as client:
        result = await client.call_tool(
            "start_training",
            {
                "dataset": DS_ID,
                "project": "warehouse-safety",
                "base_model": "yolo26n.pt",
                "epochs": 50,
                "imgsz": 640,
                "confirm_spend": True,
            },
        )
    shell_body = json.loads(shell.calls[0].request.content)
    assert shell_body == {"projectId": PROJ_ID, "task": "detect"}
    start_body = json.loads(start.calls[0].request.content)
    assert start_body == {
        "modelId": MODEL_ID,
        "gpuType": "rtx-4090",
        "trainArgs": {
            "model": "yolo26n.pt",
            "data": "ul://javier/datasets/forklifts",
            "epochs": 50,
            "imgsz": 640,
        },
    }
    data = result.data
    assert data["model_id"] == MODEL_ID
    assert data["estimated_cost"] == "$1.73"
    assert data["estimated_cost_cents"] == 173
    assert data["balance_cents"] == 2500
    assert "get_training_status" in data["note"]


async def test_start_training_insufficient_balance_cleans_up_shell(app, platform_mock):
    platform_mock.get(f"/api/datasets/{DS_ID}").mock(
        return_value=httpx.Response(
            200,
            json={"dataset": {**SAMPLE_DATASET, "_id": DS_ID, "username": "j", "slug": "s"}},
        )
    )
    platform_mock.get("/api/projects").mock(
        return_value=httpx.Response(
            200, json={"projects": [{**SAMPLE_PROJECT, "_id": PROJ_ID}], "total": 1}
        )
    )
    platform_mock.post("/api/models").mock(
        return_value=httpx.Response(201, json={"modelId": MODEL_ID, "slug": "exp"})
    )
    platform_mock.post("/api/training/start").mock(
        return_value=httpx.Response(
            402,
            json={"error": "Insufficient balance for this run.", "code": "INSUFFICIENT_BALANCE"},
        )
    )
    cleanup = platform_mock.delete(f"/api/models/{MODEL_ID}").mock(
        return_value=httpx.Response(200, json={"success": True})
    )
    async with client_for(app) as client:
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool(
                "start_training",
                {"dataset": DS_ID, "project": "warehouse-safety", "confirm_spend": True},
            )
    message = str(excinfo.value)
    assert "Insufficient balance" in message
    assert "Top up" in message
    assert cleanup.called  # no orphaned pending shell


async def test_start_training_rejects_bad_base_model(app, platform_mock):
    async with client_for(app) as client:
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool(
                "start_training",
                {
                    "dataset": DS_ID,
                    "project": PROJ_ID,
                    "base_model": "gs://bucket/model.pt.bak",
                    "confirm_spend": True,
                },
            )
    assert "ul://" in str(excinfo.value)
    assert not platform_mock.calls
