from __future__ import annotations

from ultralytics_mcp.tools.deployments import (
    create_deployment,
    delete_deployment,
    get_deployment,
    get_deployment_logs,
    list_deployments,
)


async def test_deployment_tools_map_to_public_sdk(sdk):
    await list_deployments(owner="team", status="ready")
    sdk.deployments.list.assert_awaited_once_with("team", status="ready", limit=20)
    await create_deployment("project", "model", "prod", "Production", "europe-west1", owner="team")
    sdk.deployments.create.assert_awaited_once_with(
        "team",
        project="project",
        model="model",
        deployment="prod",
        name="Production",
        region="europe-west1",
    )
    await get_deployment_logs("prod", owner="team", severity="ERROR", page_token="next")
    sdk.deployments.logs.assert_awaited_once_with(
        "team", "prod", severity="ERROR", limit=50, page_token="next"
    )


async def test_deployment_detail_preserves_three_raw_responses(sdk):
    result = await get_deployment("prod", owner="team", metrics_range="24h")
    assert result == {
        "deployment": {"deployment": {"status": "ready"}},
        "health": {"healthy": True},
        "metrics": {"summary": {}},
    }
    sdk.deployments.retrieve.assert_awaited_once_with("team", "prod")
    sdk.deployments.health.assert_awaited_once_with("team", "prod")
    sdk.deployments.metrics.assert_awaited_once_with("team", "prod", range="24h")


async def test_delete_deployment_requires_confirmation(sdk):
    result = await delete_deployment("prod", owner="team")
    assert result["confirmation_required"] is True
    sdk.deployments.delete.assert_not_awaited()
    await delete_deployment("prod", owner="team", confirm=True)
    sdk.deployments.delete.assert_awaited_once_with("team", "prod")
