"""Deployment tools (US2/US3, FR-003/FR-004): browse and manage inference endpoints.

Deployments never spend credits — plan-tier deployment counts are the only quota.
Creation and deletion are API-driven; start/stop and health/logs/metrics require
the web UI until the platform accepts API keys on those routes.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

from fastmcp.exceptions import ToolError
from pydantic import Field

from ..auth import get_request_token
from ..errors import platform_errors
from ..platform_client import platform_api
from ..schemas import (
    DeploymentStatus,
    DeploymentSummary,
    bounded_dump,
    clamp_limit,
    looks_like_object_id,
    make_list_result,
)


async def _resolve_deployment(token: str, deployment: str) -> DeploymentSummary | dict[str, Any]:
    """Resolve an id, slug or name to one deployment summary, or the candidates payload."""
    hint = f"Deployment '{deployment}'"
    if looks_like_object_id(deployment):
        detail = await platform_api.get(
            f"/api/deployments/{deployment}", token=token, resource_hint=hint
        )
        return DeploymentSummary.from_api(detail.get("deployment", {}))
    data = await platform_api.get(
        "/api/deployments",
        token=token,
        params={"limit": clamp_limit(None)},
        resource_hint=hint,
    )
    matches = [
        DeploymentSummary.from_api(item)
        for item in data.get("deployments", [])
        if deployment in (item.get("slug"), item.get("name"))
    ]
    if not matches:
        raise ToolError(
            f"Deployment '{deployment}' was not found. Use list_deployments to see what's running."
        )
    if len(matches) > 1:
        return {
            "candidates": [m.model_dump(exclude_none=True) for m in matches],
            "note": f"Multiple deployments match '{deployment}' — call again with the id.",
        }
    return matches[0]


@platform_errors
async def list_deployments(
    model_id: Annotated[
        str | None, Field(description="Filter by the deployed model's id (24-char hex)")
    ] = None,
    status: Annotated[str | None, Field(description="Filter by deployment status")] = None,
    limit: Annotated[
        int | None, Field(description="Max deployments to return (default 20, max 50)", ge=1)
    ] = None,
) -> dict[str, Any]:
    """List your dedicated inference deployments.

    Read-only — spends nothing. Returns each deployment's id, name, model, lifecycle
    status, region and service URL. Use get_deployment for health and metrics.
    """
    token = get_request_token()
    params: dict[str, Any] = {"limit": clamp_limit(limit)}
    if model_id:
        params["modelId"] = model_id
    if status:
        params["status"] = status
    data = await platform_api.get(
        "/api/deployments", token=token, params=params, resource_hint="Your deployment list"
    )
    deployments = [DeploymentSummary.from_api(item) for item in data.get("deployments", [])]
    return make_list_result(
        deployments,
        total=data.get("total"),
        empty_note="No deployments yet — create one with create_deployment.",
    )


@platform_errors
async def get_deployment(
    deployment: Annotated[str, Field(description="Deployment id (24-char hex), slug or name")],
) -> dict[str, Any]:
    """Check one deployment's status, health and performance metrics.

    Read-only — spends nothing. Returns lifecycle status, whether the endpoint is
    healthy, its latency, and request/error-rate metrics — one call answers
    "is my endpoint OK?". If a name matches several deployments, the candidates are
    returned instead of guessing.
    """
    token = get_request_token()
    hint = f"Deployment '{deployment}'"
    resolved = await _resolve_deployment(token, deployment)
    if isinstance(resolved, dict):
        return resolved
    summary = resolved

    health, metrics = await asyncio.gather(
        platform_api.get(f"/api/deployments/{summary.id}/health", token=token, resource_hint=hint),
        platform_api.get(f"/api/deployments/{summary.id}/metrics", token=token, resource_hint=hint),
        return_exceptions=True,
    )
    health_data = {} if isinstance(health, BaseException) else health
    metrics_summary = {} if isinstance(metrics, BaseException) else (metrics.get("summary") or {})
    status = DeploymentStatus(
        deployment=summary,
        healthy=health_data.get("healthy"),
        health_error=health_data.get("error"),
        latency_ms=health_data.get("latencyMs"),
        total_requests=metrics_summary.get("totalRequests"),
        error_rate=metrics_summary.get("errorRate"),
        avg_latency_ms=metrics_summary.get("avgLatencyMs"),
    )
    return bounded_dump(status)


@platform_errors
async def create_deployment(
    model: Annotated[
        str, Field(description="Model id (24-char hex) or slug — must have trained weights")
    ],
    name: Annotated[
        str, Field(description="Deployment name, e.g. 'detector-prod' (becomes the slug)")
    ],
    region: Annotated[
        str,
        Field(
            description="Cloud Run region closest to the traffic, e.g. 'us-central1', "
            "'europe-west1', 'europe-southwest1', 'asia-northeast1'"
        ),
    ],
) -> dict[str, Any]:
    """Deploy a trained model as a dedicated inference endpoint.

    State-changing — spends no credits (deployments scale to zero when idle; plan
    tiers cap how many you can have). Provisioning is asynchronous and typically
    takes 1-4 minutes: poll get_deployment until status is 'ready', then send
    predictions to its service URL with your API key.
    """
    token = get_request_token()
    from .models import _resolve_model

    resolved = await _resolve_model(token, model)
    if isinstance(resolved, dict):
        return resolved
    body = {"modelId": resolved, "name": name, "region": region}
    data = await platform_api.post(
        "/api/deployments", token=token, json=body, resource_hint=f"Deployment '{name}'"
    )
    return {
        "deployment_id": str(data.get("deploymentId", "")),
        "status": data.get("status"),
        "region": data.get("region", region),
        "note": "Provisioning started — poll get_deployment until status is 'ready' "
        "(usually 1-4 minutes).",
    }


@platform_errors
async def delete_deployment(
    deployment: Annotated[str, Field(description="Deployment id (24-char hex), slug or name")],
    confirm: Annotated[
        bool,
        Field(
            description="Must be true — deletion is PERMANENT (no trash): the endpoint "
            "and its record are removed immediately"
        ),
    ] = False,
) -> dict[str, Any]:
    """Permanently delete an inference deployment and its endpoint.

    State-changing and IRREVERSIBLE — there is no trash for deployments; the Cloud
    Run service and the record are removed immediately. Spends no credits.
    Redeploying the model later creates a new endpoint with a new URL.
    """
    if not confirm:
        raise ToolError(
            "Deleting a deployment is permanent (no trash) — its endpoint URL stops "
            "working immediately. Call again with confirm=true to proceed."
        )
    token = get_request_token()
    resolved = await _resolve_deployment(token, deployment)
    if isinstance(resolved, dict):
        return resolved
    await platform_api.delete(
        f"/api/deployments/{resolved.id}",
        token=token,
        resource_hint=f"Deployment '{deployment}'",
    )
    return {
        "success": True,
        "deployment_id": resolved.id,
        "note": "Deployment permanently deleted — the endpoint URL no longer serves.",
    }


def register(mcp) -> None:
    mcp.tool(list_deployments)
    mcp.tool(get_deployment)
    mcp.tool(create_deployment)
    mcp.tool(delete_deployment)
