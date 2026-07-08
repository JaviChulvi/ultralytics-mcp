"""US1: per-request credentials — guidance when absent/invalid, isolation, no leakage."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastmcp.exceptions import ToolError

from tests.conftest import SAMPLE_PROJECT, TEST_TOKEN, client_for
from ultralytics_mcp.settings import settings

TOKEN_A = "ul_test_user_a_token"
TOKEN_B = "ul_test_user_b_token"


async def test_missing_header_yields_credential_guidance(app, platform_mock):
    async with client_for(app, token=None) as client:
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool("list_projects", {})
    message = str(excinfo.value)
    assert "API key" in message
    assert "Settings > API Keys" in message


async def test_invalid_key_yields_credential_guidance(app, platform_mock):
    platform_mock.get("/api/projects").mock(return_value=httpx.Response(401))
    async with client_for(app) as client:
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool("list_projects", {})
    assert "Settings > API Keys" in str(excinfo.value)


async def test_token_never_echoed_in_output_or_logs(app, platform_mock, caplog):
    platform_mock.get("/api/projects").mock(
        return_value=httpx.Response(200, json={"projects": [SAMPLE_PROJECT], "total": 1})
    )
    async with client_for(app) as client:
        result = await client.call_tool("list_projects", {})
    assert TEST_TOKEN not in str(result.content)
    assert TEST_TOKEN not in caplog.text


async def test_concurrent_users_are_isolated(app, platform_mock):
    """Two users on one instance: each upstream call carries exactly its own token (FR-002)."""

    def responder(request: httpx.Request) -> httpx.Response:
        token = request.headers["authorization"].removeprefix("Bearer ")
        name = {"ul_test_user_a_token": "project-of-a", "ul_test_user_b_token": "project-of-b"}[
            token
        ]
        return httpx.Response(
            200, json={"projects": [{**SAMPLE_PROJECT, "name": name}], "total": 1}
        )

    platform_mock.get("/api/projects").mock(side_effect=responder)

    async with (
        client_for(app, token=TOKEN_A) as client_a,
        client_for(app, token=TOKEN_B) as client_b,
    ):
        result_a, result_b = await asyncio.gather(
            client_a.call_tool("list_projects", {}),
            client_b.call_tool("list_projects", {}),
        )
    assert "project-of-a" in str(result_a.content)
    assert "project-of-b" in str(result_b.content)
    assert "project-of-b" not in str(result_a.content)
    assert "project-of-a" not in str(result_b.content)


async def test_default_limit_applied_upstream(app, platform_mock):
    route = platform_mock.get("/api/projects").mock(
        return_value=httpx.Response(200, json={"projects": [], "total": 0})
    )
    async with client_for(app) as client:
        result = await client.call_tool("list_projects", {})
    sent = route.calls[0].request
    assert f"limit={settings.default_page_size}" in str(sent.url)
    assert "No projects yet" in str(result.content)
