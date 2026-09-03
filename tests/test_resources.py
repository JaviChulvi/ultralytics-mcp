from __future__ import annotations

from ultralytics_mcp.tools.account import get_account_status, search_platform
from ultralytics_mcp.tools.datasets import (
    create_dataset,
    create_dataset_version,
    get_dataset,
    get_dataset_download,
    import_dataset_from_url,
    list_datasets,
)
from ultralytics_mcp.tools.exports import create_export, get_export, list_exports
from ultralytics_mcp.tools.models import (
    get_gpu_availability,
    get_model,
    get_model_files,
    get_training_status,
    list_models,
)
from ultralytics_mcp.tools.projects import create_project, get_project, list_projects


async def test_account_composite_preserves_raw_responses(sdk):
    result = await get_account_status()
    assert result == {"summary": {"username": "test-user"}, "storage": {"usage": {}}}


async def test_search_maps_filters_and_clamps_limit(sdk):
    await search_platform(q="forklift", starred=True, limit=500)
    sdk.explore.search.assert_awaited_once_with(q="forklift", limit=50, starred="true")


async def test_project_tools_use_explicit_owner_and_sdk_methods(sdk):
    await list_projects(owner="team", limit=0)
    sdk.projects.list.assert_awaited_once_with("team", limit=1)
    await get_project("demo", owner="team", search="best")
    sdk.projects.retrieve.assert_awaited_once_with("team", "demo", search="best")
    await create_project("new-project", "New Project", owner="team", visibility="private")
    sdk.projects.create.assert_awaited_once_with(
        project="new-project", name="New Project", owner="team", visibility="private"
    )


async def test_dataset_composite_and_single_response_modes(sdk):
    await list_datasets(owner="team", limit=60, include_samples=True)
    sdk.datasets.list.assert_awaited_once_with(
        "team", limit=50, include_samples="true", include_image_urls="false"
    )

    result = await get_dataset("data", owner="team")
    assert result == {"dataset": {"dataset": {}}, "classStats": {"classes": []}}
    sdk.datasets.retrieve.assert_awaited_once_with("team", "data")
    sdk.datasets.class_stats.assert_awaited_once_with("team", "data")

    sdk.datasets.retrieve.reset_mock()
    result = await get_dataset("data", owner="team", include_stats=False)
    assert result == {"dataset": {}}
    sdk.datasets.retrieve.assert_awaited_once_with("team", "data")


async def test_dataset_mutations_build_narrow_sdk_inputs(sdk):
    await create_dataset("data", "Data", owner="team", task="detect")
    sdk.datasets.create.assert_awaited_once_with(
        dataset="data", name="Data", owner="team", task="detect"
    )
    await import_dataset_from_url(
        "data",
        "https://example.test/data.zip",
        owner="team",
        target_split="train",
        conflict_policy="skip",
    )
    sdk.datasets.ingest.assert_awaited_once_with(
        "team",
        "data",
        body={
            "sourceUrl": "https://example.test/data.zip",
            "targetSplit": "train",
            "conflictPolicy": "skip",
        },
    )
    await get_dataset_download("data", owner="team", version=2)
    sdk.datasets.export.assert_awaited_once_with("team", "data", v=2)
    await create_dataset_version("data", owner="team", description="baseline")
    sdk.datasets.create_export.assert_awaited_once_with("team", "data", description="baseline")


async def test_model_read_tools_map_to_public_sdk(sdk):
    await list_models("project", owner="team", limit=60)
    sdk.models.list.assert_awaited_once_with("team", "project", limit=50)
    await get_model("project", "model", owner="team", include_analysis=True)
    sdk.models.retrieve.assert_awaited_once_with("team", "project", "model", analysis="1")
    await get_model_files("project", "model", owner="team")
    sdk.models.files.assert_awaited_once_with("team", "project", "model")
    await get_training_status("project", "model", owner="team")
    sdk.models.training.assert_awaited_once_with("team", "project", "model")
    await get_gpu_availability()
    sdk.training.gpu_availability.assert_awaited_once_with(managed="true")


async def test_export_tools_map_to_public_sdk(sdk):
    await list_exports("project", "model", owner="team", status="running")
    sdk.exports.list.assert_awaited_once_with(
        "team", "project", "model", status="running", limit=20
    )
    await create_export("project", "model", "onnx", owner="team", args={"dynamic": True})
    sdk.exports.create.assert_awaited_once_with(
        "team", "project", "model", format="onnx", args={"dynamic": True}
    )
    await get_export("project", "model", "export-1", owner="team")
    sdk.exports.retrieve.assert_awaited_once_with("team", "project", "model", "export-1")
