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
    """The detail endpoint accepts a slug directly — one lookup, no listing round-trip."""
    platform_mock.get("/api/datasets/forklifts").mock(
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
    platform_mock.get("/api/datasets/no-such-dataset").mock(return_value=httpx.Response(404))
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
        exports = await client.call_tool(
            "list_exports", {"model_id": "a" * 24, "status": "completed"}
        )
        deployments = await client.call_tool("list_deployments", {})
    assert exports.data["items"][0]["format"] == "onnx"
    assert deployments.data["items"][0]["service_url"] == "https://dep-001.run.app"


async def test_get_dataset_exact_match_among_fuzzy_results(app, platform_mock):
    """When the direct lookup 404s, fuzzy listing results must resolve by exact match."""
    fuzzy = [
        {**SAMPLE_DATASET, "_id": "ds_other", "name": "other", "slug": "other-dataset"},
        {**SAMPLE_DATASET, "_id": "ds_002", "name": "VisDroneDET", "slug": "visdronedet"},
    ]
    platform_mock.get("/api/datasets/VisDroneDET").mock(return_value=httpx.Response(404))
    platform_mock.get("/api/datasets").mock(
        return_value=httpx.Response(200, json={"datasets": fuzzy, "total": 2})
    )
    platform_mock.get("/api/datasets/ds_002").mock(
        return_value=httpx.Response(200, json={"dataset": fuzzy[1]})
    )
    platform_mock.get("/api/datasets/ds_002/class-stats").mock(
        return_value=httpx.Response(200, json=SAMPLE_CLASS_STATS)
    )
    async with client_for(app) as client:
        result = await client.call_tool("get_dataset", {"dataset": "VisDroneDET"})
    data = result.data
    assert "candidates" not in data
    assert data["dataset"]["id"] == "ds_002"


async def test_get_dataset_fuzzy_junk_raises_not_found(app, platform_mock):
    """Nothing actually matching the ref is 'not found', never 'multiple datasets match'."""
    junk = [
        {**SAMPLE_DATASET, "_id": "ds_a"},
        {**SAMPLE_DATASET, "_id": "ds_b", "name": "unrelated", "slug": "unrelated"},
    ]
    platform_mock.get("/api/datasets/r6s-apka").mock(return_value=httpx.Response(404))
    platform_mock.get("/api/datasets").mock(
        return_value=httpx.Response(200, json={"datasets": junk, "total": 2})
    )
    async with client_for(app) as client:
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool("get_dataset", {"dataset": "r6s-apka"})
    assert "not found" in str(excinfo.value).lower()


async def test_get_dataset_multiple_exact_matches_returns_candidates(app, platform_mock):
    twins = [{**SAMPLE_DATASET, "_id": "ds_001"}, {**SAMPLE_DATASET, "_id": "ds_002"}]
    platform_mock.get("/api/datasets/forklifts").mock(return_value=httpx.Response(404))
    platform_mock.get("/api/datasets").mock(
        return_value=httpx.Response(200, json={"datasets": twins, "total": 2})
    )
    async with client_for(app) as client:
        result = await client.call_tool("get_dataset", {"dataset": "forklifts"})
    data = result.data
    assert "dataset" not in data  # never a silent pick (US2 sc3)
    assert len(data["candidates"]) == 2
    assert "exact id" in data["note"]


async def test_get_dataset_owner_slug_ref_direct_lookup(app, platform_mock):
    """'owner/datasets/slug' hits the detail endpoint directly with the username param."""
    target = {**SAMPLE_DATASET, "_id": "ds_003", "name": "R6s apka", "slug": "r6s-apka"}
    direct = platform_mock.get("/api/datasets/r6s-apka", params={"username": "austin-smith"}).mock(
        return_value=httpx.Response(200, json={"dataset": target})
    )
    platform_mock.get("/api/datasets/ds_003/class-stats").mock(
        return_value=httpx.Response(200, json=SAMPLE_CLASS_STATS)
    )
    async with client_for(app) as client:
        result = await client.call_tool(
            "get_dataset", {"dataset": "austin-smith/datasets/r6s-apka"}
        )
    assert result.data["dataset"]["id"] == "ds_003"
    assert direct.called


async def test_get_dataset_owner_slug_ref_falls_back_to_listing(app, platform_mock):
    """If the direct slug lookup fails (live API 400s on slugs), the owner's list resolves."""
    target = {**SAMPLE_DATASET, "_id": "ds_003", "name": "R6s apka", "slug": "r6s-apka"}
    platform_mock.get("/api/datasets/r6s-apka").mock(return_value=httpx.Response(400))
    listing = platform_mock.get("/api/datasets", params={"username": "austin-smith"}).mock(
        return_value=httpx.Response(200, json={"datasets": [SAMPLE_DATASET, target], "total": 2})
    )
    platform_mock.get("/api/datasets/ds_003", params={"username": "austin-smith"}).mock(
        return_value=httpx.Response(200, json={"dataset": target})
    )
    platform_mock.get("/api/datasets/ds_003/class-stats").mock(
        return_value=httpx.Response(200, json=SAMPLE_CLASS_STATS)
    )
    async with client_for(app) as client:
        result = await client.call_tool(
            "get_dataset", {"dataset": "austin-smith/datasets/r6s-apka"}
        )
    assert result.data["dataset"]["id"] == "ds_003"
    assert "slug=" not in str(listing.calls[0].request.url)


async def test_list_datasets_username_slug_falls_back_to_local_match(app, platform_mock):
    """The upstream slug+username combination returns nothing for datasets that exist."""
    target = {**SAMPLE_DATASET, "_id": "ds_003", "name": "R6s apka", "slug": "r6s-apka"}
    platform_mock.get(
        "/api/datasets", params={"slug": "r6s-apka", "username": "austin-smith"}
    ).mock(return_value=httpx.Response(200, json={"datasets": [], "total": 0}))
    platform_mock.get("/api/datasets", params={"username": "austin-smith"}).mock(
        return_value=httpx.Response(200, json={"datasets": [SAMPLE_DATASET, target], "total": 2})
    )
    async with client_for(app) as client:
        result = await client.call_tool(
            "list_datasets", {"slug": "r6s-apka", "username": "austin-smith"}
        )
    data = result.data
    assert data["returned"] == 1
    assert data["items"][0]["slug"] == "r6s-apka"


async def test_list_datasets_empty_with_filters_names_the_filters(app, platform_mock):
    platform_mock.get("/api/datasets").mock(
        return_value=httpx.Response(200, json={"datasets": [], "total": 0})
    )
    async with client_for(app) as client:
        result = await client.call_tool("list_datasets", {"username": "ghost"})
    assert "No datasets matched" in result.data["note"]


async def test_list_datasets_long_class_names_truncated(app, platform_mock):
    big = {**SAMPLE_DATASET, "classNames": [f"c{i}" for i in range(80)], "classCount": 80}
    platform_mock.get("/api/datasets").mock(
        return_value=httpx.Response(200, json={"datasets": [big], "total": 1})
    )
    async with client_for(app) as client:
        result = await client.call_tool("list_datasets", {})
    item = result.data["items"][0]
    assert len(item["class_names"]) == 15
    assert item["class_names_omitted"] == 65
    assert item["class_count"] == 80


async def test_get_dataset_names_classes_beyond_preview(app, platform_mock):
    """Class stats must resolve names past the class_names preview truncation."""
    dataset_id = "e" * 24
    names = [f"c{i}" for i in range(80)]
    big = {**SAMPLE_DATASET, "_id": dataset_id, "classNames": names, "classCount": 80}
    platform_mock.get(f"/api/datasets/{dataset_id}").mock(
        return_value=httpx.Response(200, json={"dataset": big})
    )
    platform_mock.get(f"/api/datasets/{dataset_id}/class-stats").mock(
        return_value=httpx.Response(
            200,
            json={"classes": [{"classId": 40, "count": 7, "imageCount": 5}], "sampled": False},
        )
    )
    async with client_for(app) as client:
        result = await client.call_tool("get_dataset", {"dataset": dataset_id})
    assert result.data["classes"][0]["name"] == "c40"


def _image(width: int, height: int, label_count: int) -> dict:
    return {
        "hash": f"h{width}x{height}",
        "name": f"img-{width}x{height}.jpg",
        "split": "train",
        "width": width,
        "height": height,
        "labelCount": label_count,
    }


async def test_get_dataset_image_stats(app, platform_mock):
    dataset_id = "f" * 24
    detail = {
        **SAMPLE_DATASET,
        "_id": dataset_id,
        "splits": {"train": 3, "val": 1, "test": 0, "labeled": 3},
    }
    platform_mock.get(f"/api/datasets/{dataset_id}").mock(
        return_value=httpx.Response(200, json={"dataset": detail})
    )
    platform_mock.get(f"/api/datasets/{dataset_id}/class-stats").mock(
        return_value=httpx.Response(200, json=SAMPLE_CLASS_STATS)
    )
    images_url = f"/api/datasets/{dataset_id}/images"
    # hasLabel routes first: respx params-matching is subset-based, first match wins.
    platform_mock.get(images_url, params={"split": "train", "hasLabel": "false"}).mock(
        return_value=httpx.Response(200, json={"images": [], "total": 1})
    )
    platform_mock.get(images_url, params={"split": "val", "hasLabel": "false"}).mock(
        return_value=httpx.Response(200, json={"images": [], "total": 0})
    )
    platform_mock.get(images_url, params={"split": "train"}).mock(
        return_value=httpx.Response(
            200,
            json={
                "images": [_image(640, 480, 3), _image(1920, 1080, 8), _image(640, 480, 0)],
                "total": 3,
            },
        )
    )
    platform_mock.get(images_url, params={"split": "val"}).mock(
        return_value=httpx.Response(
            200, json={"images": [_image(640, 640, 2)], "total": 1, "errorCount": 1}
        )
    )
    async with client_for(app) as client:
        result = await client.call_tool(
            "get_dataset", {"dataset": dataset_id, "include_image_stats": True}
        )
    data = result.data
    train = data["image_stats"]["splits"]["train"]
    assert train["images"] == 3
    assert train["unlabeled_images"] == 1
    assert train["labels_per_image"] == {"min": 0, "median": 3, "max": 8, "sample_size": 3}
    assert train["dimensions"]["smallest"] == "640x480"
    assert train["dimensions"]["largest"] == "1920x1080"
    assert "error_images" not in train
    assert data["image_stats"]["splits"]["val"]["error_images"] == 1
    assert "test" not in data["image_stats"]["splits"]  # empty split not probed
    assert data["image_stats_note"]
    sample_call = platform_mock.get(images_url, params={"split": "train"}).calls[0]
    assert "includeThumbnails=false" in str(sample_call.request.url)


async def test_get_dataset_overall_stats_from_class_stats_histograms(app, platform_mock):
    """Whole-dataset distributions ride along with class-stats — no extra requests."""
    dataset_id = "a" * 24
    platform_mock.get(f"/api/datasets/{dataset_id}").mock(
        return_value=httpx.Response(200, json={"dataset": {**SAMPLE_DATASET, "_id": dataset_id}})
    )
    platform_mock.get(f"/api/datasets/{dataset_id}/class-stats").mock(
        return_value=httpx.Response(
            200,
            json={
                **SAMPLE_CLASS_STATS,
                "sampled": True,
                "sampleSize": 1000,
                "imageStats": {
                    "objectsPerImageHistogram": [
                        {"bin": 0, "count": 2, "size": 1},
                        {"bin": 1, "count": 5, "size": 1},
                        {"bin": 3, "count": 3, "size": 1},
                    ],
                    "widthHistogram": [
                        {"bin": 640, "count": 9, "size": 1},
                        {"bin": 1920, "count": 1, "size": 1},
                    ],
                    "heightHistogram": [
                        {"bin": 480, "count": 9, "size": 1},
                        {"bin": 1080, "count": 1, "size": 1},
                    ],
                    "formatDistribution": {"jpg": 10},
                },
            },
        )
    )
    async with client_for(app) as client:
        result = await client.call_tool("get_dataset", {"dataset": dataset_id})
    data = result.data
    overall = data["image_stats"]["overall"]
    assert overall["unlabeled_images"] == 2
    assert overall["labels_per_image"] == {"min": 0, "median": 1, "max": 3}
    assert overall["width"] == {"min": 640, "max": 1920}
    assert overall["height"] == {"min": 480, "max": 1080}
    assert overall["formats"] == {"jpg": 10}
    assert "splits" not in data["image_stats"]  # only with include_image_stats
    assert data["stats_sampled"] is True
    assert data["stats_sample_size"] == 1000


async def test_get_dataset_overall_stats_ranged_zero_bin_omits_unlabeled(app, platform_mock):
    """A ranged 0-bin can't separate unlabeled images — the count must be absent, not 0."""
    dataset_id = "b" * 24
    platform_mock.get(f"/api/datasets/{dataset_id}").mock(
        return_value=httpx.Response(200, json={"dataset": {**SAMPLE_DATASET, "_id": dataset_id}})
    )
    platform_mock.get(f"/api/datasets/{dataset_id}/class-stats").mock(
        return_value=httpx.Response(
            200,
            json={
                **SAMPLE_CLASS_STATS,
                "imageStats": {
                    "objectsPerImageHistogram": [
                        {"bin": 0, "count": 4300, "size": 50},
                        {"bin": 900, "count": 12, "size": 50},
                    ]
                },
            },
        )
    )
    async with client_for(app) as client:
        result = await client.call_tool("get_dataset", {"dataset": dataset_id})
    overall = result.data["image_stats"]["overall"]
    assert "unlabeled_images" not in overall
    assert overall["labels_per_image"]["max"] == 949


async def test_list_dataset_images_fields_filter(app, platform_mock):
    dataset_id = "d" * 24
    platform_mock.get(f"/api/datasets/{dataset_id}/images").mock(
        return_value=httpx.Response(200, json={"images": [_image(640, 480, 3)], "total": 1})
    )
    async with client_for(app) as client:
        result = await client.call_tool(
            "list_dataset_images", {"dataset": dataset_id, "fields": ["name", "split"]}
        )
    assert set(result.data["items"][0]) == {"name", "split"}


async def test_list_dataset_images_unknown_field_rejected(app, platform_mock):
    async with client_for(app) as client:
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool("list_dataset_images", {"dataset": "d" * 24, "fields": ["nope"]})
    assert "nope" in str(excinfo.value)


async def test_list_dataset_images_has_label_param(app, platform_mock):
    dataset_id = "d" * 24
    route = platform_mock.get(f"/api/datasets/{dataset_id}/images").mock(
        return_value=httpx.Response(200, json={"images": [], "total": 0})
    )
    async with client_for(app) as client:
        await client.call_tool("list_dataset_images", {"dataset": dataset_id, "has_label": False})
    assert "hasLabel=false" in str(route.calls[0].request.url)


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
