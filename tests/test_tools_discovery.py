"""Public catalog search and user profiles."""

from __future__ import annotations

import httpx
import pytest
from fastmcp.exceptions import ToolError

from tests.conftest import client_for

EXPLORE_DATASET = {
    "_id": "pub_ds_1",
    "slug": "wildfire-smoke",
    "name": "Wildfire Smoke",
    "description": "Aerial smoke plumes for early wildfire detection.",
    "username": "forest-watch",
    "visibility": "public",
    "imageCount": 4800,
    "classCount": 1,
    "classNames": ["smoke"],
    "task": "detect",
    "starCount": 12,
    "updatedAt": "2026-06-01T00:00:00Z",
}

EXPLORE_PROJECT = {
    "_id": "pub_pr_1",
    "slug": "yolo11",
    "name": "YOLO11",
    "description": "Official YOLO11 models.",
    "username": "ultralytics",
    "modelCount": 25,
    "modelNames": ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x", "yolo11n-seg"],
    "starCount": 900,
    "updatedAt": "2026-07-01T00:00:00Z",
}


async def test_search_platform_merges_types_and_flags_more(app, platform_mock):
    route = platform_mock.get("/api/explore/search").mock(
        return_value=httpx.Response(
            200,
            json={"projects": [EXPLORE_PROJECT], "datasets": [EXPLORE_DATASET], "hasMore": True},
        )
    )
    async with client_for(app) as client:
        result = await client.call_tool("search_platform", {"query": "smoke"})
    data = result.data
    assert {i["type"] for i in data["items"]} == {"project", "dataset"}
    dataset = next(i for i in data["items"] if i["type"] == "dataset")
    assert dataset["username"] == "forest-watch"
    assert dataset["class_names"] == ["smoke"]
    project = next(i for i in data["items"] if i["type"] == "project")
    assert len(project["model_names"]) == 5  # capped preview
    assert "offset=20" in data["note"]
    url = str(route.calls[0].request.url)
    assert "q=smoke" in url and "type=all" in url


async def test_search_platform_passes_filters(app, platform_mock):
    route = platform_mock.get("/api/explore/search").mock(
        return_value=httpx.Response(200, json={"projects": [], "datasets": [], "hasMore": False})
    )
    async with client_for(app) as client:
        result = await client.call_tool(
            "search_platform",
            {
                "query": "coco",
                "kind": "datasets",
                "sort": "count-desc",
                "task": "detect,segment",
                "author": "ultralytics",
                "offset": 20,
            },
        )
    url = str(route.calls[0].request.url)
    for fragment in (
        "type=datasets",
        "sort=count-desc",
        "task=detect%2Csegment",
        "author=ultralytics",
        "offset=20",
    ):
        assert fragment in url
    assert "No public results" in result.data["note"]


@pytest.mark.parametrize(
    ("args", "fragment"),
    [
        ({"kind": "models"}, "kind"),
        ({"sort": "alphabetical"}, "sort"),
    ],
)
async def test_search_platform_rejects_bad_enums_before_calling(app, platform_mock, args, fragment):
    async with client_for(app) as client:
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool("search_platform", {"query": "x", **args})
    assert fragment in str(excinfo.value).lower()
    assert not platform_mock.calls


async def test_search_platform_previews_long_class_lists(app, platform_mock):
    big = {**EXPLORE_DATASET, "classNames": [f"c{i}" for i in range(80)], "classCount": 80}
    platform_mock.get("/api/explore/search").mock(
        return_value=httpx.Response(200, json={"projects": [], "datasets": [big]})
    )
    async with client_for(app) as client:
        result = await client.call_tool("search_platform", {"query": "c", "kind": "datasets"})
    item = result.data["items"][0]
    assert len(item["class_names"]) == 15
    assert item["class_names_omitted"] == 65


async def test_get_user_profile(app, platform_mock):
    platform_mock.get("/api/users", params={"username": "forest-watch"}).mock(
        return_value=httpx.Response(
            200,
            json={
                "user": {
                    "username": "forest-watch",
                    "fullName": "Forest Watch",
                    "accountType": "team",
                    "bio": "Wildfire early-warning research.",
                    "company": "FW Labs",
                    "followerCount": 31,
                    "socials": {"github": "https://github.com/forest-watch", "twitter": None},
                }
            },
        )
    )
    async with client_for(app) as client:
        result = await client.call_tool("get_user_profile", {"username": "forest-watch"})
    user = result.data["user"]
    assert user["full_name"] == "Forest Watch"
    assert user["socials"] == {"github": "https://github.com/forest-watch"}  # empties dropped


async def test_get_user_profile_not_found_names_username(app, platform_mock):
    platform_mock.get("/api/users").mock(return_value=httpx.Response(404))
    async with client_for(app) as client:
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool("get_user_profile", {"username": "ghost"})
    assert "ghost" in str(excinfo.value)
