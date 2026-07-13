"""Dataset lifecycle: downloads, versions, lineage, create/update/trash, URL import."""

from __future__ import annotations

import json

import httpx
import pytest
from fastmcp.exceptions import ToolError

from tests.conftest import SAMPLE_MODEL, client_for

DS_ID = "a" * 24


async def test_get_dataset_download_current(app, platform_mock):
    platform_mock.get(f"/api/datasets/{DS_ID}/export").mock(
        return_value=httpx.Response(
            200, json={"downloadUrl": "https://gcs/x.ndjson", "cached": True}
        )
    )
    async with client_for(app) as client:
        result = await client.call_tool("get_dataset_download", {"dataset": DS_ID})
    data = result.data
    assert data["download_url"] == "https://gcs/x.ndjson"
    assert data["format"] == "ndjson"
    assert "7 days" in data["note"]


async def test_get_dataset_download_version_param(app, platform_mock):
    route = platform_mock.get(f"/api/datasets/{DS_ID}/export", params={"v": 3}).mock(
        return_value=httpx.Response(
            200, json={"downloadUrl": "https://gcs/v3.ndjson", "version": 3}
        )
    )
    async with client_for(app) as client:
        result = await client.call_tool("get_dataset_download", {"dataset": DS_ID, "version": 3})
    assert route.called
    assert result.data["version"] == 3


async def test_get_dataset_download_processing_conflict_surfaces_reason(app, platform_mock):
    platform_mock.get(f"/api/datasets/{DS_ID}/export").mock(
        return_value=httpx.Response(409, json={"error": "Dataset is processing an import."})
    )
    async with client_for(app) as client:
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool("get_dataset_download", {"dataset": DS_ID})
    assert "processing an import" in str(excinfo.value)


async def test_create_dataset_version(app, platform_mock):
    route = platform_mock.post(f"/api/datasets/{DS_ID}/export").mock(
        return_value=httpx.Response(200, json={"version": 2, "downloadUrl": "https://gcs/v2"})
    )
    async with client_for(app) as client:
        result = await client.call_tool(
            "create_dataset_version", {"dataset": DS_ID, "description": "before relabeling"}
        )
    assert result.data["version"] == 2
    assert json.loads(route.calls[0].request.content) == {"description": "before relabeling"}


async def test_list_dataset_models_enriches_project_context(app, platform_mock):
    platform_mock.get(f"/api/datasets/{DS_ID}/models").mock(
        return_value=httpx.Response(
            200,
            json={
                "models": [{**SAMPLE_MODEL, "projectSlug": "warehouse", "username": "javier"}],
                "count": 1,
            },
        )
    )
    async with client_for(app) as client:
        result = await client.call_tool("list_dataset_models", {"dataset": DS_ID})
    item = result.data["items"][0]
    assert item["project_slug"] == "warehouse"
    assert item["username"] == "javier"
    assert result.data["total"] == 1


async def test_list_dataset_models_empty_note(app, platform_mock):
    platform_mock.get(f"/api/datasets/{DS_ID}/models").mock(
        return_value=httpx.Response(200, json={"models": [], "count": 0})
    )
    async with client_for(app) as client:
        result = await client.call_tool("list_dataset_models", {"dataset": DS_ID})
    assert "No models" in result.data["note"]


async def test_create_dataset_derives_slug_and_parses_response(app, platform_mock):
    route = platform_mock.post("/api/datasets").mock(
        return_value=httpx.Response(201, json={"datasetId": DS_ID, "slug": "river-litter-2"})
    )
    async with client_for(app) as client:
        result = await client.call_tool(
            "create_dataset",
            {"name": "River Litter!", "task": "segment", "class_names": ["bottle", "bag"]},
        )
    body = json.loads(route.calls[0].request.content)
    assert body == {
        "name": "River Litter!",
        "slug": "river-litter",
        "task": "segment",
        "classNames": ["bottle", "bag"],
    }
    assert result.data["dataset_id"] == DS_ID
    assert result.data["slug"] == "river-litter-2"  # platform de-duped


@pytest.mark.parametrize(
    ("args", "fragment"),
    [
        ({"name": "x", "task": "recognize"}, "task"),
        ({"name": "x", "visibility": "hidden"}, "visibility"),
        ({"name": "!!!"}, "letter or number"),
    ],
)
async def test_create_dataset_validates_before_calling(app, platform_mock, args, fragment):
    async with client_for(app) as client:
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool("create_dataset", args)
    assert fragment in str(excinfo.value).lower()
    assert not platform_mock.calls


async def test_update_dataset_sends_only_given_fields(app, platform_mock):
    route = platform_mock.patch(f"/api/datasets/{DS_ID}").mock(
        return_value=httpx.Response(200, json={"success": True, "slug": "new-name"})
    )
    async with client_for(app) as client:
        result = await client.call_tool(
            "update_dataset", {"dataset": DS_ID, "name": "New Name", "visibility": "public"}
        )
    assert json.loads(route.calls[0].request.content) == {
        "name": "New Name",
        "visibility": "public",
    }
    assert result.data == {"success": True, "slug": "new-name"}


async def test_update_dataset_requires_a_field(app, platform_mock):
    async with client_for(app) as client:
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool("update_dataset", {"dataset": DS_ID})
    assert "at least one" in str(excinfo.value)
    assert not platform_mock.calls


async def test_delete_dataset_soft_deletes_with_recovery_note(app, platform_mock):
    route = platform_mock.delete(f"/api/datasets/{DS_ID}").mock(
        return_value=httpx.Response(200, json={"success": True})
    )
    async with client_for(app) as client:
        result = await client.call_tool("delete_dataset", {"dataset": DS_ID})
    assert route.called
    assert result.data["success"] is True
    assert "restore_from_trash" in result.data["note"]
    assert "30 days" in result.data["note"]


async def test_import_into_existing_dataset(app, platform_mock):
    route = platform_mock.post("/api/datasets/ingest").mock(
        return_value=httpx.Response(
            201, json={"jobId": "job_1", "datasetId": DS_ID, "status": "queued"}
        )
    )
    async with client_for(app) as client:
        result = await client.call_tool(
            "import_dataset_from_url",
            {
                "source_url": "https://example.com/data.zip",
                "dataset": DS_ID,
                "target_split": "train",
            },
        )
    assert json.loads(route.calls[0].request.content) == {
        "datasetId": DS_ID,
        "sourceUrl": "https://example.com/data.zip",
        "targetSplit": "train",
    }
    data = result.data
    assert data["job_id"] == "job_1"
    assert data["status"] == "queued"
    assert "poll get_dataset" in data["note"]


async def test_import_creates_dataset_when_named(app, platform_mock):
    platform_mock.post("/api/datasets").mock(
        return_value=httpx.Response(201, json={"datasetId": DS_ID, "slug": "aerial-sheep"})
    )
    platform_mock.post("/api/datasets/ingest").mock(
        return_value=httpx.Response(
            201, json={"jobId": "job_2", "datasetId": DS_ID, "status": "queued"}
        )
    )
    async with client_for(app) as client:
        result = await client.call_tool(
            "import_dataset_from_url",
            {"source_url": "https://example.com/sheep.tar.gz", "name": "Aerial Sheep"},
        )
    data = result.data
    assert data["dataset_id"] == DS_ID
    assert data["slug"] == "aerial-sheep"
    assert data["job_id"] == "job_2"


@pytest.mark.parametrize(
    "args",
    [
        {"source_url": "https://x/y.zip"},  # neither
        {"source_url": "https://x/y.zip", "dataset": DS_ID, "name": "both"},  # both
    ],
)
async def test_import_requires_exactly_one_target(app, platform_mock, args):
    async with client_for(app) as client:
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool("import_dataset_from_url", args)
    assert "exactly one" in str(excinfo.value)
    assert not platform_mock.calls


async def test_import_conflict_names_inflight_job(app, platform_mock):
    platform_mock.post("/api/datasets/ingest").mock(
        return_value=httpx.Response(
            409, json={"error": "Dataset is already processing.", "existingJobId": "job_9"}
        )
    )
    async with client_for(app) as client:
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool(
                "import_dataset_from_url",
                {"source_url": "https://x/y.zip", "dataset": DS_ID},
            )
    message = str(excinfo.value)
    assert "already processing" in message
    assert "job_9" in message
