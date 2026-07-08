"""Model tools (US2/US3, FR-003/FR-004). All read-only."""

from __future__ import annotations

from typing import Annotated, Any

from fastmcp.exceptions import ToolError
from pydantic import Field

from ..auth import get_request_token
from ..errors import platform_errors
from ..platform_client import platform_api
from ..schemas import (
    ModelSummary,
    TrainingStatus,
    clamp_limit,
    looks_like_object_id,
    make_list_result,
)


@platform_errors
async def list_models(
    project: Annotated[str, Field(description="Project id (24-char hex) or slug")],
    username: Annotated[
        str | None,
        Field(description="Project owner's username when browsing another user's public project"),
    ] = None,
    limit: Annotated[
        int | None, Field(description="Max models to return (default 20, max 50)", ge=1)
    ] = None,
) -> dict[str, Any]:
    """List the models in a project.

    Read-only — spends nothing. Returns each model's id, name, task, training status,
    epochs and best fitness. Use get_model for details or get_training_status for a
    live run.
    """
    token = get_request_token()
    params: dict[str, Any] = {"limit": clamp_limit(limit)}
    if looks_like_object_id(project):
        params["projectId"] = project
    else:
        params["projectSlug"] = project
    if username:
        params["username"] = username
    data = await platform_api.get(
        "/api/models", token=token, params=params, resource_hint=f"Models of '{project}'"
    )
    models = [ModelSummary.from_api(item) for item in data.get("models", [])]
    return make_list_result(models, empty_note=f"No models in project '{project}' yet.")


@platform_errors
async def get_model(
    model: Annotated[str, Field(description="Model id (24-char hex) or slug")],
    project: Annotated[
        str | None, Field(description="Project id or slug, to disambiguate a model slug")
    ] = None,
    username: Annotated[str | None, Field(description="Owner username, if not your own")] = None,
) -> dict[str, Any]:
    """Get one model's details by id or slug.

    Read-only — spends nothing. Returns name, task, training status, epochs, best
    epoch and best fitness. If a slug matches several models, the candidates are
    returned instead of guessing.
    """
    token = get_request_token()
    hint = f"Model '{model}'"
    if looks_like_object_id(model):
        data = await platform_api.get(f"/api/models/{model}", token=token, resource_hint=hint)
        summary = ModelSummary.from_api(data.get("model", {}))
        return {"model": summary.model_dump(exclude_none=True)}
    params: dict[str, Any] = {"slug": model, "limit": 5}
    if project:
        if looks_like_object_id(project):
            params["projectId"] = project
        else:
            params["projectSlug"] = project
    if username:
        params["username"] = username
    data = await platform_api.get("/api/models", token=token, params=params, resource_hint=hint)
    matches = [ModelSummary.from_api(item) for item in data.get("models", [])]
    if not matches:
        raise ToolError(
            f"Model '{model}' was not found. Use list_models to see a project's models."
        )
    if len(matches) > 1:
        return {
            "candidates": [m.model_dump(exclude_none=True) for m in matches],
            "note": f"Multiple models match '{model}' — call again with the exact id.",
        }
    return {"model": matches[0].model_dump(exclude_none=True)}


@platform_errors
async def get_training_status(
    model: Annotated[str, Field(description="Model id (24-char hex) or slug")],
) -> dict[str, Any]:
    """Check a model's live training progress: state, epochs, latest metrics.

    Read-only — spends nothing. For a model that is training, returns the current
    epoch, completion percentage and the latest metric values; for a model that never
    trained, says so plainly.
    """
    token = get_request_token()
    hint = f"Model '{model}'"
    model_id = model
    if not looks_like_object_id(model):
        data = await platform_api.get(
            "/api/models", token=token, params={"slug": model, "limit": 5}, resource_hint=hint
        )
        matches = [ModelSummary.from_api(item) for item in data.get("models", [])]
        if not matches:
            raise ToolError(f"Model '{model}' was not found. Use list_models to find it.")
        if len(matches) > 1:
            return {
                "candidates": [m.model_dump(exclude_none=True) for m in matches],
                "note": f"Multiple models match '{model}' — call again with the exact id.",
            }
        model_id = matches[0].id
    data = await platform_api.get(
        f"/api/models/{model_id}/training", token=token, resource_hint=hint
    )
    status = TrainingStatus.from_api(model_id, data)
    if not data.get("trainResults"):
        status.note = (
            "This model has no training history — it hasn't started training yet, "
            "so there is no progress to report."
        )
    return status.model_dump(exclude_none=True)


def register(mcp) -> None:
    mcp.tool(list_models)
    mcp.tool(get_model)
    mcp.tool(get_training_status)
