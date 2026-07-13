"""Opt-in live smoke tests against the real platform API.

Excluded by default (pytest addopts `-m 'not live'`). Run with:
    ULTRALYTICS_TEST_API_KEY=ul_... uv run pytest -m live
"""

from __future__ import annotations

import os

import pytest

from tests.conftest import client_for

KEY = os.environ.get("ULTRALYTICS_TEST_API_KEY")

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not KEY, reason="ULTRALYTICS_TEST_API_KEY not set"),
]


async def test_projects_and_models_live(app):
    async with client_for(app, token=KEY) as client:
        projects = (await client.call_tool("list_projects", {"limit": 3})).data
        assert "items" in projects
        if projects["items"]:
            models = (
                await client.call_tool("list_models", {"project": projects["items"][0]["id"]})
            ).data
            assert "items" in models


async def test_datasets_live(app):
    async with client_for(app, token=KEY) as client:
        datasets = (await client.call_tool("list_datasets", {"limit": 3})).data
        assert "items" in datasets


async def test_exports_and_deployments_live(app):
    async with client_for(app, token=KEY) as client:
        deployments = (await client.call_tool("list_deployments", {"limit": 3})).data
        assert "items" in deployments
        # exports need a model id; only check when the account has one
        projects = (await client.call_tool("list_projects", {"limit": 1})).data
        if projects["items"]:
            models = (
                await client.call_tool("list_models", {"project": projects["items"][0]["id"]})
            ).data
            if models["items"]:
                exports = (
                    await client.call_tool("list_exports", {"model_id": models["items"][0]["id"]})
                ).data
                assert "items" in exports


async def test_dataset_download_and_lineage_live(app):
    async with client_for(app, token=KEY) as client:
        datasets = (await client.call_tool("list_datasets", {"limit": 1})).data
        if not datasets["items"]:
            pytest.skip("account has no datasets")
        ds_id = datasets["items"][0]["id"]
        download = (await client.call_tool("get_dataset_download", {"dataset": ds_id})).data
        assert download["download_url"].startswith("http")
        lineage = (await client.call_tool("list_dataset_models", {"dataset": ds_id})).data
        assert "items" in lineage


async def test_discovery_live(app):
    async with client_for(app, token=KEY) as client:
        search = (await client.call_tool("search_platform", {"query": "coco"})).data
        assert "items" in search
        if search["items"]:
            username = search["items"][0].get("username")
            if username:
                profile = (await client.call_tool("get_user_profile", {"username": username})).data
                assert profile["user"].get("username") == username


@pytest.mark.xfail(
    reason="platform currently rejects API keys on billing/storage/activity endpoints "
    "(2026-07-08); raised with the platform team",
    strict=False,
)
async def test_account_live(app):
    async with client_for(app, token=KEY) as client:
        status = (await client.call_tool("get_account_status", {})).data
        activity = (await client.call_tool("get_recent_activity", {"limit": 5})).data
        assert "credits_cents" in status or "plan" in status
        assert "items" in activity
