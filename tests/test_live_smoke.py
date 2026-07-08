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
