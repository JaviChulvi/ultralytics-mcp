"""Managed deployment tools backed directly by the official SDK."""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from ..errors import sdk_errors
from ..runtime import runtime
from .common import provided


@sdk_errors
async def list_deployments(
    owner: str | None = None,
    status: Literal["creating", "deploying", "ready", "stopping", "stopped", "failed"]
    | None = None,
    model: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """List managed deployment endpoints. Read-only and spends nothing."""
    return await runtime.sdk().deployments.list(
        await runtime.owner(owner),
        **provided(status=status, model=model, limit=max(1, min(limit, 50))),
    )


@sdk_errors
async def create_deployment(
    project: str,
    model: str,
    deployment: str,
    name: str,
    region: str,
    owner: str | None = None,
) -> dict[str, Any]:
    """Create a managed inference endpoint. State-changing but spends no credits.

    The endpoint provisions asynchronously. Poll get_deployment until it becomes
    ready or failed.
    """
    resolved_owner = await runtime.owner(owner)
    return await runtime.sdk().deployments.create(
        resolved_owner,
        project=project,
        model=model,
        deployment=deployment,
        name=name,
        region=region,
    )


@sdk_errors
async def get_deployment(
    deployment: str,
    owner: str | None = None,
    metrics_range: Literal["1h", "6h", "24h", "7d", "30d"] = "1h",
) -> dict[str, Any]:
    """Get deployment details, health, and metrics in one read-only call."""
    resolved_owner = await runtime.owner(owner)
    client = runtime.sdk()
    detail, health, metrics = await asyncio.gather(
        client.deployments.retrieve(resolved_owner, deployment),
        client.deployments.health(resolved_owner, deployment),
        client.deployments.metrics(resolved_owner, deployment, range=metrics_range),
    )
    return {"deployment": detail, "health": health, "metrics": metrics}


@sdk_errors
async def get_deployment_logs(
    deployment: str,
    owner: str | None = None,
    severity: str | None = None,
    limit: int = 50,
    page_token: str | None = None,
) -> dict[str, Any]:
    """Get recent deployment logs. Read-only and spends nothing."""
    return await runtime.sdk().deployments.logs(
        await runtime.owner(owner),
        deployment,
        **provided(
            severity=severity,
            limit=max(1, min(limit, 100)),
            page_token=page_token,
        ),
    )


@sdk_errors
async def delete_deployment(
    deployment: str, owner: str | None = None, confirm: bool = False
) -> dict[str, Any]:
    """Permanently delete a managed deployment.

    Deployments do not enter trash. This tool does nothing unless confirm=true follows
    the user's explicit request.
    """
    if not confirm:
        return {
            "deleted": False,
            "confirmation_required": True,
            "message": "Deployment deletion is permanent; retry with confirm=true to proceed.",
        }
    return await runtime.sdk().deployments.delete(await runtime.owner(owner), deployment)


def register(mcp) -> None:
    for tool in (
        list_deployments,
        create_deployment,
        get_deployment,
        get_deployment_logs,
        delete_deployment,
    ):
        mcp.tool(tool)
