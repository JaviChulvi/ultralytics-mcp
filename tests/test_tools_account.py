"""US4: account insight — balance+storage merge, activity feed."""

from __future__ import annotations

import httpx

from tests.conftest import client_for


async def test_account_status_merges_balance_and_storage(app, platform_mock):
    balance = platform_mock.get("/api/billing/balance").mock(
        return_value=httpx.Response(200, json={"creditsCents": 12345, "plan": "pro"})
    )
    storage = platform_mock.get("/api/storage").mock(
        return_value=httpx.Response(
            200, json={"tier": "standard", "usage": {"storage": 5_368_709_120}}
        )
    )
    async with client_for(app) as client:
        result = await client.call_tool("get_account_status", {})
    data = result.data
    assert data["credits_cents"] == 12345
    assert data["plan"] == "pro"
    assert data["storage_tier"] == "standard"
    assert data["storage_used_bytes"] == 5_368_709_120
    assert balance.called and storage.called


async def test_recent_activity_summarized_and_paged(app, platform_mock):
    route = platform_mock.get("/api/activity").mock(
        return_value=httpx.Response(
            200,
            json={
                "events": [
                    {
                        "action": "model.trained",
                        "resourceType": "model",
                        "resourceName": "detector-v1",
                        "timestamp": "2026-07-07T09:00:00Z",
                        "userEmail": "someone@example.com",
                        "metadata": {"internal": "stuff"},
                    }
                ],
                "total": 31,
                "page": 2,
            },
        )
    )
    async with client_for(app) as client:
        result = await client.call_tool("get_recent_activity", {"page": 2})
    data = result.data
    assert "page=2" in str(route.calls[0].request.url)
    item = data["items"][0]
    assert item["action"] == "model.trained"
    assert item["resource_name"] == "detector-v1"
    assert "userEmail" not in item  # whitelist shaping: internals never pass through
    assert data["total"] == 31
    assert data["truncated"] is True


async def test_empty_activity_feed_says_so(app, platform_mock):
    platform_mock.get("/api/activity").mock(
        return_value=httpx.Response(200, json={"events": [], "total": 0})
    )
    async with client_for(app) as client:
        result = await client.call_tool("get_recent_activity", {})
    assert "No recent activity" in result.data["note"]
