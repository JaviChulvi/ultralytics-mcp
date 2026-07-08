"""Shared fixtures: respx-mocked platform API + in-process MCP client over real HTTP.

The MCP client drives the actual ASGI app through httpx.ASGITransport, so the
Authorization header travels exactly as it would in production; respx intercepts only
the outbound httpx traffic to the platform (default transport), never the test client.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from asgi_lifespan import LifespanManager
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from ultralytics_mcp.server import create_app
from ultralytics_mcp.settings import settings

TEST_TOKEN = "ul_test_0123456789abcdef"  # never a real key

SAMPLE_PROJECT = {
    "_id": "proj_001",
    "name": "warehouse-safety",
    "slug": "warehouse-safety",
    "visibility": "private",
    "modelCount": 2,
    "updatedAt": "2026-07-01T10:00:00Z",
}

SAMPLE_DATASET = {
    "_id": "ds_001",
    "name": "forklifts",
    "slug": "forklifts",
    "task": "detect",
    "visibility": "private",
    "imageCount": 1200,
    "classCount": 3,
    "classNames": ["forklift", "person", "pallet"],
    "splits": {"train": 1000, "val": 150, "test": 50},
}

SAMPLE_MODEL = {
    "_id": "mod_001",
    "name": "detector-v1",
    "slug": "detector-v1",
    "projectId": "proj_001",
    "task": "detect",
    "status": "trained",
    "epochs": 50,
    "bestEpoch": 42,
    "bestFitness": 0.71,
}

SAMPLE_DEPLOYMENT = {
    "_id": "dep_001",
    "name": "prod-endpoint",
    "modelId": "mod_001",
    "status": "running",
    "statusMessage": "Serving",
    "region": "us-central1",
    "serviceUrl": "https://dep-001.run.app",
}

SAMPLE_EXPORT = {
    "_id": "exp_001",
    "modelId": "mod_001",
    "format": "onnx",
    "status": "completed",
    "file": "model.onnx",
    "completedAt": "2026-07-02T12:00:00Z",
}


@pytest.fixture
def platform_mock():
    with respx.mock(base_url=settings.platform_base_url, assert_all_called=False) as mock:
        yield mock


@pytest.fixture
async def app():
    application = create_app()
    async with LifespanManager(application) as manager:
        yield manager.app


def client_for(app, token: str | None = TEST_TOKEN) -> Client:
    """MCP client wired to the ASGI app; token=None sends no Authorization header."""
    headers = {"Authorization": f"Bearer {token}"} if token is not None else {}

    def factory(
        headers: dict | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
        **kwargs,
    ) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
            headers=headers,
            timeout=timeout,
            auth=auth,
            **kwargs,
        )

    transport = StreamableHttpTransport(
        url="http://testserver/mcp", headers=headers, httpx_client_factory=factory
    )
    return Client(transport)


@pytest.fixture
async def mcp_client(app):
    async with client_for(app) as client:
        yield client
