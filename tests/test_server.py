from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastmcp import Client

import ultralytics_mcp.runtime as runtime_module
from ultralytics_mcp.runtime import CREDENTIAL_GUIDANCE, PlatformRuntime, runtime
from ultralytics_mcp.server import mcp


async def test_registers_exact_v1_surface():
    tools = await mcp.list_tools()
    assert {tool.name for tool in tools} == {
        "get_account_status",
        "search_platform",
        "list_projects",
        "get_project",
        "create_project",
        "list_datasets",
        "get_dataset",
        "create_dataset",
        "import_dataset_from_url",
        "get_dataset_download",
        "create_dataset_version",
        "list_models",
        "get_model",
        "get_model_files",
        "get_training_status",
        "get_gpu_availability",
        "start_training",
        "cancel_training",
        "list_exports",
        "create_export",
        "get_export",
        "list_deployments",
        "create_deployment",
        "get_deployment",
        "get_deployment_logs",
        "delete_deployment",
    }


async def test_runtime_requires_environment_key(monkeypatch):
    monkeypatch.delenv("ULTRALYTICS_API_KEY", raising=False)
    instance = PlatformRuntime()
    with pytest.raises(RuntimeError, match="ULTRALYTICS_API_KEY"):
        await instance.start()
    assert "api-keys" in CREDENTIAL_GUIDANCE


async def test_runtime_owns_and_closes_sdk(monkeypatch):
    client = AsyncMock()
    monkeypatch.setenv("ULTRALYTICS_API_KEY", "ul_test")
    monkeypatch.setattr(runtime_module, "AsyncPlatform", lambda: client)
    instance = PlatformRuntime()
    await instance.start()
    assert instance.client is client
    await instance.close()
    client.close.assert_awaited_once_with()
    assert instance.client is None


async def test_owner_is_resolved_once(sdk):
    runtime._default_owner = None
    assert await runtime.owner() == "test-user"
    assert await runtime.owner() == "test-user"
    sdk.account.summary.assert_awaited_once_with()
    assert await runtime.owner("team-owner") == "team-owner"


async def test_in_memory_stdio_protocol_executes_registered_tool(sdk, monkeypatch):
    monkeypatch.setattr(runtime, "start", AsyncMock())
    monkeypatch.setattr(runtime, "close", AsyncMock())
    async with Client(mcp) as client:
        result = await client.call_tool("get_account_status", {})
    assert result.data == {
        "summary": {"username": "test-user"},
        "storage": {"usage": {}},
    }
