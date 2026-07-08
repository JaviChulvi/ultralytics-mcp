"""Export and deployment tools (US2/US3, FR-003/FR-004). All read-only."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import Field

from ..auth import get_request_token
from ..errors import platform_errors
from ..platform_client import platform_api
from ..schemas import DeploymentSummary, ExportSummary, clamp_limit, make_list_result


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


def register(mcp) -> None:
    mcp.tool(list_exports)
    mcp.tool(list_deployments)
