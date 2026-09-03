from __future__ import annotations

import os

import pytest
from ultralytics_platform import AsyncPlatform


@pytest.mark.live
async def test_read_only_sdk_smoke():
    api_key = os.environ.get("ULTRALYTICS_TEST_API_KEY")
    if not api_key:
        pytest.skip("ULTRALYTICS_TEST_API_KEY is not set")
    async with AsyncPlatform(api_key=api_key) as client:
        summary = await client.account.summary()
        assert summary["username"]
        await client.explore.search(limit=1)
        await client.projects.list(summary["username"], limit=1)
        await client.datasets.list(summary["username"], limit=1)
