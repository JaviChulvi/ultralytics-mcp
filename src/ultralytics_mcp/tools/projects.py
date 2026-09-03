"""Project tools backed directly by the official SDK."""

from __future__ import annotations

from typing import Any, Literal

from ..errors import sdk_errors
from ..runtime import runtime
from .common import provided


@sdk_errors
async def list_projects(owner: str | None = None, limit: int = 20) -> dict[str, Any]:
    """List projects in a personal or team workspace. Read-only and spends nothing."""
    return await runtime.sdk().projects.list(
        await runtime.owner(owner), limit=max(1, min(limit, 50))
    )


@sdk_errors
async def get_project(
    project: str, owner: str | None = None, search: str | None = None
) -> dict[str, Any]:
    """Get one project and its model summaries by owner and project slug.

    Read-only and spends nothing.
    """
    return await runtime.sdk().projects.retrieve(
        await runtime.owner(owner), project, **provided(search=search)
    )


@sdk_errors
async def create_project(
    project: str,
    name: str,
    owner: str | None = None,
    description: str | None = None,
    visibility: Literal["public", "private"] | None = None,
    tags: list[str] | None = None,
    license: str | None = None,
) -> dict[str, Any]:
    """Create a project. State-changing but does not spend training credits.

    `project` is the lowercase URL slug; `name` is the display name. Set owner for a
    team workspace, otherwise the authenticated personal workspace is used.
    """
    resolved_owner = await runtime.owner(owner)
    return await runtime.sdk().projects.create(
        **provided(
            project=project,
            name=name,
            owner=resolved_owner,
            description=description,
            visibility=visibility,
            tags=tags,
            license=license,
        )
    )


def register(mcp) -> None:
    mcp.tool(list_projects)
    mcp.tool(get_project)
    mcp.tool(create_project)
