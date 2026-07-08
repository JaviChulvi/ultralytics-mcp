"""US2: browse/inspect tools — shaping, paging caps, empty states, disambiguation."""

from __future__ import annotations

import json

import httpx
import pytest
from fastmcp.exceptions import ToolError

from tests.conftest import (
    SAMPLE_DATASET,
    SAMPLE_DEPLOYMENT,
    SAMPLE_EXPORT,
    SAMPLE_MODEL,
    SAMPLE_PROJECT,
    client_for,
)
from ultralytics_mcp.settings import settings

SAMPLE_CLASS_STATS = {
    "classes": [
        {"classId": 0, "count": 5000, "imageCount": 900},
        {"classId": 1, "count": 2400, "imageCount": 700},
        {"classId": 2, "count": 800, "imageCount": 400},
    ],
    "classNames": ["forklift", "person", "pallet"],
    "sampled": False,
}


async def test_list_datasets_happy_path(app, platform_mock):
    platform_mock.get("/api/datasets").mock(
        return_value=httpx.Response(200, json={"datasets": [SAMPLE_DATASET], "total": 1})
    )
    async with client_for(app) as client:
        result = await client.call_tool("list_datasets", {})
    data = result.data
    assert data["returned"] == 1
    assert data["items"][0]["name"] == "forklifts"
    assert data["items"][0]["image_count"] == 1200
    assert not data["truncated"]


async def test_get_dataset_by_slug_includes_class_stats(app, platform_mock):
    platform_mock.get("/api/datasets", params={"slug": "forklifts"}).mock(
        return_value=httpx.Response(200, json={"datasets": [SAMPLE_DATASET], "total": 1})
    )
    platform_mock.get("/api/datasets/ds_001").mock(
        return_value=httpx.Response(200, json={"dataset": SAMPLE_DATASET})
    )
    platform_mock.get("/api/datasets/ds_001/class-stats").mock(
        return_value=httpx.Response(200, json=SAMPLE_CLASS_STATS)
    )
    async with client_for(app) as client:
        result = await client.call_tool("get_dataset", {"dataset": "forklifts"})
    data = result.data
    assert data["dataset"]["name"] == "forklifts"
    by_name = {c["name"]: c for c in data["classes"]}
    assert by_name["forklift"]["instance_count"] == 5000
    assert by_name["pallet"]["image_count"] == 400


async def test_get_dataset_unknown_slug_names_what_was_looked_up(app, platform_mock):
    platform_mock.get("/api/datasets").mock(
        return_value=httpx.Response(200, json={"datasets": [], "total": 0})
    )
    async with client_for(app) as client:
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool("get_dataset", {"dataset": "no-such-dataset"})
    message = str(excinfo.value)
    assert "no-such-dataset" in message
    assert "not found" in message.lower()


async def test_get_project_multiple_matches_returns_candidates(app, platform_mock):
    twins = [
        {**SAMPLE_PROJECT, "_id": "proj_001", "username": "alice"},
        {**SAMPLE_PROJECT, "_id": "proj_002", "username": "bob"},
    ]
    platform_mock.get("/api/projects").mock(
        return_value=httpx.Response(200, json={"projects": twins, "total": 2})
    )
    async with client_for(app) as client:
        result = await client.call_tool("get_project", {"project": "warehouse-safety"})
    data = result.data
    assert "project" not in data  # never a silent pick (US2 sc3)
    assert len(data["candidates"]) == 2
    assert "exact id" in data["note"]


async def test_limit_clamped_to_max(app, platform_mock):
    route = platform_mock.get("/api/datasets").mock(
        return_value=httpx.Response(200, json={"datasets": [], "total": 0})
    )
    async with client_for(app) as client:
        await client.call_tool("list_datasets", {"limit": 500})
    assert f"limit={settings.max_page_size}" in str(route.calls[0].request.url)


async def test_empty_account_says_so_plainly(app, platform_mock):
    platform_mock.get("/api/datasets").mock(
        return_value=httpx.Response(200, json={"datasets": [], "total": 0})
    )
    async with client_for(app) as client:
        result = await client.call_tool("list_datasets", {})
    assert "No datasets yet" in result.data["note"]


async def test_max_page_stays_under_size_cap(app, platform_mock):
    """SC-005: a max-size page serializes below the cap, with the truncation flag set."""
    big = [
        {
            **SAMPLE_DATASET,
            "_id": f"ds_{i:03d}",
            "name": f"a-rather-long-dataset-name-for-size-testing-{i:03d}",
            "classNames": [f"class-name-{j}" for j in range(20)],
        }
        for i in range(50)
    ]
    platform_mock.get("/api/datasets").mock(
        return_value=httpx.Response(200, json={"datasets": big, "total": 500})
    )
    async with client_for(app) as client:
        result = await client.call_tool("list_datasets", {"limit": 50})
    data = result.data
    assert len(json.dumps(data)) <= settings.max_response_bytes
    assert data["truncated"] is True
    assert data["note"]


async def test_list_models_sends_project_slug(app, platform_mock):
    route = platform_mock.get("/api/models").mock(
        return_value=httpx.Response(200, json={"models": [SAMPLE_MODEL]})
    )
    async with client_for(app) as client:
        result = await client.call_tool("list_models", {"project": "warehouse-safety"})
    assert "projectSlug=warehouse-safety" in str(route.calls[0].request.url)
    assert result.data["items"][0]["best_fitness"] == 0.71


async def test_get_model_by_id(app, platform_mock):
    platform_mock.get(f"/api/models/{'a' * 24}").mock(
        return_value=httpx.Response(200, json={"model": SAMPLE_MODEL})
    )
    async with client_for(app) as client:
        result = await client.call_tool("get_model", {"model": "a" * 24})
    assert result.data["model"]["status"] == "trained"


async def test_list_exports_and_deployments(app, platform_mock):
    platform_mock.get("/api/exports").mock(
        return_value=httpx.Response(200, json={"exports": [SAMPLE_EXPORT]})
    )
    platform_mock.get("/api/deployments").mock(
        return_value=httpx.Response(200, json={"deployments": [SAMPLE_DEPLOYMENT], "total": 1})
    )
    async with client_for(app) as client:
        exports = await client.call_tool("list_exports", {"status": "completed"})
        deployments = await client.call_tool("list_deployments", {})
    assert exports.data["items"][0]["format"] == "onnx"
    assert deployments.data["items"][0]["service_url"] == "https://dep-001.run.app"


async def test_list_dataset_images_paging(app, platform_mock):
    dataset_id = "d" * 24
    platform_mock.get(f"/api/datasets/{dataset_id}/images").mock(
        return_value=httpx.Response(
            200,
            json={
                "images": [
                    {
                        "hash": "h1",
                        "name": "img1.jpg",
                        "split": "train",
                        "width": 640,
                        "height": 480,
                        "labelCount": 3,
                    }
                ],
                "total": 1200,
                "hasMore": True,
                "nextCursor": "cursor-xyz",
            },
        )
    )
    async with client_for(app) as client:
        result = await client.call_tool(
            "list_dataset_images", {"dataset": dataset_id, "split": "train"}
        )
    data = result.data
    assert data["has_more"] is True
    assert data["next_cursor"] == "cursor-xyz"
    assert data["items"][0]["label_count"] == 3
