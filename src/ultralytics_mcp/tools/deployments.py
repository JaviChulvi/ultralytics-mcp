"""Export and deployment tools (US2/US3, FR-003/FR-004). All read-only."""

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
    ExportSummary,
    bounded_dump,
    clamp_limit,
    looks_like_object_id,
    make_list_result,
)


@platform_errors
async def list_exports(
    model_id: Annotated[
        str | None, Field(description="Filter by the source model's id (24-char hex)")
    ] = None,
    status: Annotated[str | None, Field(description="Filter by export status")] = None,
    limit: Annotated[
        int | None, Field(description="Max exports to return (default 20, max 50)", ge=1)
    ] = None,
) -> dict[str, Any]:
    """List your model exports (format conversions like ONNX, TensorRT, CoreML).

    Read-only — spends nothing. Returns each export's id, source model, format,
    status, artifact file name and completion time.
    """
    token = get_request_token()
    params: dict[str, Any] = {"limit": clamp_limit(limit)}
    if model_id:
        params["modelId"] = model_id
    if status:
        params["status"] = status
    data = await platform_api.get(
        "/api/exports", token=token, params=params, resource_hint="Your export list"
    )
    exports = [ExportSummary.from_api(item) for item in data.get("exports", [])]
    return make_list_result(
        exports, empty_note="No exports yet — export a trained model from the platform."
    )


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
        empty_note="No deployments yet — deploy a trained model from the platform.",
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
    if looks_like_object_id(deployment):
        detail = await platform_api.get(
            f"/api/deployments/{deployment}", token=token, resource_hint=hint
        )
        summary = DeploymentSummary.from_api(detail.get("deployment", {}))
    else:
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
                f"Deployment '{deployment}' was not found. "
                "Use list_deployments to see what's running."
            )
        if len(matches) > 1:
            return {
                "candidates": [m.model_dump(exclude_none=True) for m in matches],
                "note": f"Multiple deployments match '{deployment}' — call again with the id.",
            }
        summary = matches[0]

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


def register(mcp) -> None:
    mcp.tool(list_exports)
    mcp.tool(list_deployments)
    mcp.tool(get_deployment)
