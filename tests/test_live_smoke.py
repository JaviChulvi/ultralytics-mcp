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


async def test_export_cycle_live(app):
    """Create a real export job, poll it once, then remove it (self-cleaning)."""
    from fastmcp.exceptions import ToolError

    async with client_for(app, token=KEY) as client:
        projects = (await client.call_tool("list_projects", {"limit": 5})).data
        model_id = None
        for project in projects["items"]:
            models = (await client.call_tool("list_models", {"project": project["id"]})).data
            completed = [m for m in models["items"] if m.get("status") == "completed"]
            if completed:
                model_id = completed[0]["id"]
                break
        if not model_id:
            pytest.skip("account has no completed model to export")
        try:
            created = (
                await client.call_tool("create_export", {"model": model_id, "format": "onnx"})
            ).data
        except ToolError as exc:
            if "already" in str(exc):
                pytest.skip("an onnx export is already in flight on this model")
            raise
        export_id = created["export_id"]
        polled = (await client.call_tool("get_export", {"export_id": export_id})).data
        assert polled["status"] in ("queued", "starting", "running", "completed", "failed")
        removed = (await client.call_tool("delete_export", {"export_id": export_id})).data
        assert removed["action"] in ("cancelled", "deleted")


async def test_dataset_write_cycle_live(app):
    """Full write cycle against the real platform: create → rename → trash →
    restore → trash → purge. Ends with nothing left behind."""
    async with client_for(app, token=KEY) as client:
        created = (
            await client.call_tool("create_dataset", {"name": "mcp-smoke-cycle", "task": "detect"})
        ).data
        ds_id = created["dataset_id"]
        try:
            renamed = (
                await client.call_tool(
                    "update_dataset", {"dataset": ds_id, "description": "smoke test artifact"}
                )
            ).data
            assert renamed["success"] is True
            trashed = (await client.call_tool("delete_dataset", {"dataset": ds_id})).data
            assert trashed["success"] is True
            trash = (await client.call_tool("list_trash", {"type": "dataset"})).data
            assert any(i["id"] == ds_id for i in trash["items"])
            restored = (
                await client.call_tool("restore_from_trash", {"item_id": ds_id, "type": "dataset"})
            ).data
            assert restored["success"] is True
        finally:
            # Leave no trace: trash (idempotent even if already trashed) then purge.
            try:
                await client.call_tool("delete_dataset", {"dataset": ds_id})
            except Exception:
                pass
            purged = (
                await client.call_tool(
                    "purge_from_trash", {"item_id": ds_id, "type": "dataset", "confirm": True}
                )
            ).data
            assert purged["success"] is True


async def test_training_gate_live(app):
    """GPU stock is readable and the spend gate holds — without billing anything."""
    from fastmcp.exceptions import ToolError

    async with client_for(app, token=KEY) as client:
        gpus = (await client.call_tool("get_gpu_availability", {})).data
        assert any(g["id"] == "rtx-4090" for g in gpus["gpus"])
        datasets = (await client.call_tool("list_datasets", {"limit": 1})).data
        projects = (await client.call_tool("list_projects", {"limit": 1})).data
        if not (datasets["items"] and projects["items"]):
            pytest.skip("account lacks a dataset/project pair")
        with pytest.raises(ToolError, match="SPENDS CREDITS"):
            await client.call_tool(
                "start_training",
                {
                    "dataset": datasets["items"][0]["id"],
                    "project": projects["items"][0]["id"],
                },
            )


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
