"""US3: monitoring tools — training progress and deployment health/metrics."""

from __future__ import annotations

import httpx
import pytest
from fastmcp.exceptions import ToolError

from tests.conftest import SAMPLE_DEPLOYMENT, client_for

MODEL_ID = "a" * 24
DEP_ID = "b" * 24

EPOCH_METRICS = [
    {"epoch": i, "loss": round(1.0 - i * 0.015, 3), "mAP50": round(0.3 + i * 0.008, 3)}
    for i in range(20)
]


async def test_training_in_progress(app, platform_mock):
    platform_mock.get(f"/api/models/{MODEL_ID}/training").mock(
        return_value=httpx.Response(
            200, json={"status": "training", "epochs": 50, "trainResults": EPOCH_METRICS}
        )
    )
    async with client_for(app) as client:
        result = await client.call_tool("get_training_status", {"model": MODEL_ID})
    data = result.data
    assert data["status"] == "training"
    assert data["epochs_completed"] == 20
    assert data["epochs_total"] == 50
    assert data["progress_percent"] == 40.0
    assert data["latest_metrics"]["epoch"] == 19


async def test_never_trained_says_not_applicable(app, platform_mock):
    platform_mock.get(f"/api/models/{MODEL_ID}/training").mock(
        return_value=httpx.Response(200, json={"status": "new", "trainResults": []})
    )
    async with client_for(app) as client:
        result = await client.call_tool("get_training_status", {"model": MODEL_ID})
    assert "hasn't started training" in result.data["note"]


async def test_training_completed(app, platform_mock):
    results = EPOCH_METRICS + [{"epoch": i, "loss": 0.2, "mAP50": 0.6} for i in range(20, 50)]
    platform_mock.get(f"/api/models/{MODEL_ID}/training").mock(
        return_value=httpx.Response(
            200, json={"status": "trained", "epochs": 50, "trainResults": results}
        )
    )
    async with client_for(app) as client:
        result = await client.call_tool("get_training_status", {"model": MODEL_ID})
    assert result.data["progress_percent"] == 100.0
    assert result.data["status"] == "trained"


async def test_training_status_unknown_model(app, platform_mock):
    platform_mock.get("/api/models").mock(return_value=httpx.Response(200, json={"models": []}))
    async with client_for(app) as client:
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool("get_training_status", {"model": "ghost-model"})
    assert "ghost-model" in str(excinfo.value)


async def test_deployment_healthy_composite(app, platform_mock):
    detail = platform_mock.get(f"/api/deployments/{DEP_ID}").mock(
        return_value=httpx.Response(200, json={"deployment": {**SAMPLE_DEPLOYMENT, "_id": DEP_ID}})
    )
    health = platform_mock.get(f"/api/deployments/{DEP_ID}/health").mock(
        return_value=httpx.Response(200, json={"healthy": True, "latencyMs": 42.5})
    )
    metrics = platform_mock.get(f"/api/deployments/{DEP_ID}/metrics").mock(
        return_value=httpx.Response(
            200,
            json={
                "deploymentId": DEP_ID,
                "summary": {"totalRequests": 1234, "errorRate": 0.01, "avgLatencyMs": 55.0},
            },
        )
    )
    async with client_for(app) as client:
        result = await client.call_tool("get_deployment", {"deployment": DEP_ID})
    data = result.data
    assert data["healthy"] is True
    assert data["latency_ms"] == 42.5
    assert data["total_requests"] == 1234
    assert data["deployment"]["status"] == "running"
    # composite hit all three upstream endpoints
    assert detail.called and health.called and metrics.called


async def test_deployment_stopped_and_unhealthy(app, platform_mock):
    stopped = {**SAMPLE_DEPLOYMENT, "_id": DEP_ID, "status": "stopped", "statusMessage": "Stopped"}
    platform_mock.get(f"/api/deployments/{DEP_ID}").mock(
        return_value=httpx.Response(200, json={"deployment": stopped})
    )
    platform_mock.get(f"/api/deployments/{DEP_ID}/health").mock(
        return_value=httpx.Response(200, json={"healthy": False, "error": "connection refused"})
    )
    platform_mock.get(f"/api/deployments/{DEP_ID}/metrics").mock(return_value=httpx.Response(503))
    async with client_for(app) as client:
        result = await client.call_tool("get_deployment", {"deployment": DEP_ID})
    data = result.data
    assert data["deployment"]["status"] == "stopped"
    assert data["healthy"] is False
    assert data["health_error"] == "connection refused"
    assert "total_requests" not in data  # metrics degraded gracefully, not fatally


async def test_deployment_by_name_resolution(app, platform_mock):
    platform_mock.get("/api/deployments").mock(
        return_value=httpx.Response(
            200, json={"deployments": [{**SAMPLE_DEPLOYMENT, "_id": DEP_ID}], "total": 1}
        )
    )
    platform_mock.get(f"/api/deployments/{DEP_ID}/health").mock(
        return_value=httpx.Response(200, json={"healthy": True, "latencyMs": 10.0})
    )
    platform_mock.get(f"/api/deployments/{DEP_ID}/metrics").mock(
        return_value=httpx.Response(200, json={"summary": {"totalRequests": 7}})
    )
    async with client_for(app) as client:
        result = await client.call_tool("get_deployment", {"deployment": "prod-endpoint"})
    assert result.data["deployment"]["id"] == DEP_ID
    assert result.data["healthy"] is True
