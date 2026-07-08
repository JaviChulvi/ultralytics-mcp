"""Project tools (US1/US2, FR-003). All read-only."""

from __future__ import annotations

from typing import Annotated, Any

from fastmcp.exceptions import ToolError
from pydantic import Field

from ..auth import get_request_token
from ..errors import platform_errors
from ..platform_client import platform_api
from ..schemas import ProjectSummary, clamp_limit, looks_like_object_id, make_list_result


@platform_errors
async def list_projects(
    limit: Annotated[
        int | None, Field(description="Max projects to return (default 20, max 50)", ge=1)
    ] = None,
    slug: Annotated[str | None, Field(description="Filter by exact project slug")] = None,
    username: Annotated[
        str | None,
        Field(description="Owner username — set it to browse another user's public projects"),
    ] = None,
) -> dict[str, Any]:
    """List the projects your Ultralytics Platform account can see.

    Read-only — spends nothing. Returns each project's id, name, slug, visibility,
    model count and last update, plus the account total. Narrow with slug/username;
    use get_project for one project's details.
    """
    token = get_request_token()
    params: dict[str, Any] = {"limit": clamp_limit(limit)}
    if slug:
        params["slug"] = slug
    if username:
        params["username"] = username
    data = await platform_api.get(
        "/api/projects", token=token, params=params, resource_hint="Your project list"
    )
    projects = [ProjectSummary.from_api(item) for item in data.get("projects", [])]
    return make_list_result(
        projects,
        total=data.get("total"),
        empty_note="No projects yet — create one at platform.ultralytics.com.",
    )


@platform_errors
async def get_project(
    project: Annotated[str, Field(description="Project id (24-char hex) or slug")],
    username: Annotated[
        str | None,
        Field(description="Owner username when referencing another user's public project"),
    ] = None,
) -> dict[str, Any]:
    """Get one project's details by id or slug.

    Read-only — spends nothing. Returns name, slug, visibility, model count and last
    update. If a slug matches several projects, the candidates are returned instead of
    guessing — call again with the exact id.
    """
    token = get_request_token()
    hint = f"Project '{project}'"
    if looks_like_object_id(project):
        data = await platform_api.get(f"/api/projects/{project}", token=token, resource_hint=hint)
        summary = ProjectSummary.from_api(data.get("project", {}))
        return {"project": summary.model_dump(exclude_none=True)}
    params: dict[str, Any] = {"slug": project, "limit": 5}
    if username:
        params["username"] = username
    data = await platform_api.get("/api/projects", token=token, params=params, resource_hint=hint)
    matches = [ProjectSummary.from_api(item) for item in data.get("projects", [])]
    if not matches:
        raise ToolError(
            f"Project '{project}' was not found. Use list_projects to see what's available."
        )
    if len(matches) > 1:
        return {
            "candidates": [m.model_dump(exclude_none=True) for m in matches],
            "note": f"Multiple projects match '{project}' — call again with the exact id.",
        }
    return {"project": matches[0].model_dump(exclude_none=True)}


def register(mcp) -> None:
    mcp.tool(list_projects)
    mcp.tool(get_project)
