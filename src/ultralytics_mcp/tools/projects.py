"""Project tools (US1/US2, FR-003): browse and manage projects."""

from __future__ import annotations

from typing import Annotated, Any

from fastmcp.exceptions import ToolError
from pydantic import Field

from ..auth import get_request_token
from ..errors import platform_errors
from ..platform_client import platform_api
from ..schemas import (
    ProjectSummary,
    clamp_limit,
    looks_like_object_id,
    make_list_result,
    slugify,
)

VISIBILITIES = ("public", "private")


async def _resolve_project(token: str, project: str, username: str | None = None) -> str | dict:
    """Resolve a project id or slug to its id, or return the candidates payload."""
    if looks_like_object_id(project):
        return project
    params: dict[str, Any] = {"slug": project, "limit": 5}
    if username:
        params["username"] = username
    data = await platform_api.get(
        "/api/projects", token=token, params=params, resource_hint=f"Project '{project}'"
    )
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
    return matches[0].id


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


@platform_errors
async def create_project(
    name: Annotated[str, Field(description="Human-readable project name")],
    description: Annotated[str | None, Field(description="What the project is for")] = None,
    visibility: Annotated[
        str | None, Field(description="'public' or 'private' (platform default: private)")
    ] = None,
    owner: Annotated[
        str | None, Field(description="Team username, to create in a team workspace")
    ] = None,
) -> dict[str, Any]:
    """Create a project (the container models are trained into).

    State-changing — spends no credits. The slug is derived from the name and
    de-duplicated by the platform. Use the returned id as start_training's project.
    """
    if visibility is not None and visibility not in VISIBILITIES:
        raise ToolError(f"visibility must be one of: {', '.join(VISIBILITIES)}.")
    try:
        slug = slugify(name)
    except ValueError:
        raise ToolError("The name must contain at least one letter or number.") from None
    token = get_request_token()
    body: dict[str, Any] = {"name": name, "slug": slug}
    if description:
        body["description"] = description
    if visibility:
        body["visibility"] = visibility
    if owner:
        body["owner"] = owner
    data = await platform_api.post(
        "/api/projects", token=token, json=body, resource_hint=f"Project '{name}'"
    )
    return {"project_id": str(data.get("projectId", "")), "slug": data.get("slug")}


@platform_errors
async def update_project(
    project: Annotated[str, Field(description="Project id (24-char hex) or slug")],
    name: Annotated[str | None, Field(description="New name (the slug changes with it)")] = None,
    description: Annotated[str | None, Field(description="New description")] = None,
    visibility: Annotated[str | None, Field(description="'public' or 'private'")] = None,
) -> dict[str, Any]:
    """Rename or edit a project's metadata.

    State-changing — spends no credits. Updates only the fields you pass; renaming
    re-slugs the project and keeps its models' URLs consistent.
    """
    if visibility is not None and visibility not in VISIBILITIES:
        raise ToolError(f"visibility must be one of: {', '.join(VISIBILITIES)}.")
    updates: dict[str, Any] = {}
    if name is not None:
        updates["name"] = name
    if description is not None:
        updates["description"] = description
    if visibility is not None:
        updates["visibility"] = visibility
    if not updates:
        raise ToolError("Nothing to update — pass at least one of name/description/visibility.")
    token = get_request_token()
    resolved = await _resolve_project(token, project)
    if isinstance(resolved, dict):
        return resolved
    data = await platform_api.patch(
        f"/api/projects/{resolved}",
        token=token,
        json=updates,
        resource_hint=f"Project '{project}'",
    )
    return {"success": bool(data.get("success")), "slug": data.get("slug")}


@platform_errors
async def delete_project(
    project: Annotated[str, Field(description="Project id (24-char hex) or slug")],
) -> dict[str, Any]:
    """Move a project and all its models to the trash (soft delete).

    State-changing — spends no credits. Recoverable for 30 days with
    restore_from_trash (models come back with it); after that a daily cleanup
    removes it permanently.
    """
    token = get_request_token()
    resolved = await _resolve_project(token, project)
    if isinstance(resolved, dict):
        return resolved
    await platform_api.delete(
        f"/api/projects/{resolved}", token=token, resource_hint=f"Project '{project}'"
    )
    return {
        "success": True,
        "project_id": resolved,
        "note": "Project and its models moved to trash — recoverable for 30 days "
        "with restore_from_trash.",
    }


def register(mcp) -> None:
    mcp.tool(list_projects)
    mcp.tool(get_project)
    mcp.tool(create_project)
    mcp.tool(update_project)
    mcp.tool(delete_project)
