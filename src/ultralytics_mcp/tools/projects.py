"""Project tools (US1/US2, FR-003). All read-only."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import Field

from ..auth import get_request_token
from ..errors import platform_errors
from ..platform_client import platform_api
from ..schemas import ProjectSummary, clamp_limit, make_list_result


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


def register(mcp) -> None:
    mcp.tool(list_projects)
