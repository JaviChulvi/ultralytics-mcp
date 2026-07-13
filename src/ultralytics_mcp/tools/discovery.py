"""Public discovery tools: catalog search and user profiles. All read-only."""

from __future__ import annotations

from typing import Annotated, Any

from fastmcp.exceptions import ToolError
from pydantic import Field

from ..auth import get_request_token
from ..errors import platform_errors
from ..platform_client import platform_api
from ..schemas import ExploreItem, ListResult, UserProfile, bounded_dump

SEARCH_KINDS = ("all", "projects", "datasets")
SEARCH_SORTS = ("stars", "newest", "oldest", "name-asc", "name-desc", "count-desc", "count-asc")
SEARCH_PAGE_SIZE = 20  # fixed upstream; offset is the only paging control


@platform_errors
async def search_platform(
    query: Annotated[
        str,
        Field(
            description="ONE distinctive keyword (e.g. 'wildfire', not 'wildfire smoke "
            "detection') — matching is a literal substring over names, descriptions, "
            "tags and class names, so phrases miss. Empty string browses everything."
        ),
    ] = "",
    kind: Annotated[
        str, Field(description="What to search: 'all', 'projects' or 'datasets'")
    ] = "all",
    sort: Annotated[
        str | None,
        Field(
            description="Order: stars, newest, oldest, name-asc, name-desc, "
            "count-desc or count-asc (count = images/models)"
        ),
    ] = None,
    task: Annotated[
        str | None,
        Field(
            description="Datasets only — comma-separated YOLO tasks to keep, e.g. 'detect,segment'"
        ),
    ] = None,
    author: Annotated[str | None, Field(description="Only results by this username")] = None,
    offset: Annotated[
        int | None, Field(description="Skip this many results (pages are 20 wide)", ge=0)
    ] = None,
) -> dict[str, Any]:
    """Search the public Ultralytics Platform catalog for datasets and projects.

    Read-only — spends nothing. The go-to tool for "find me a dataset/model for X":
    returns public items with owner, task, image/model counts, class names and stars.
    Verify relevance via the returned class_names — a match may be an incidental
    class-name hit. Reference results elsewhere as 'username/slug'.
    """
    if kind not in SEARCH_KINDS:
        raise ToolError(f"Unknown kind '{kind}' — choose from: {', '.join(SEARCH_KINDS)}.")
    if sort is not None and sort not in SEARCH_SORTS:
        raise ToolError(f"Unknown sort '{sort}' — choose from: {', '.join(SEARCH_SORTS)}.")
    token = get_request_token()
    params: dict[str, Any] = {"q": query, "type": kind}
    if sort:
        params["sort"] = sort
    if task:
        params["task"] = task
    if author:
        params["author"] = author
    if offset:
        params["offset"] = offset
    data = await platform_api.get(
        "/api/explore/search", token=token, params=params, resource_hint="The public catalog"
    )
    items = [ExploreItem.from_dataset(d) for d in data.get("datasets", [])] + [
        ExploreItem.from_project(p) for p in data.get("projects", [])
    ]
    dumped = [i.model_dump(exclude_none=True, mode="json") for i in items]
    note = None
    if not dumped:
        note = (
            "No public results — try a shorter, more distinctive single keyword, "
            "or drop the task/author filters."
        )
    elif data.get("hasMore"):
        note = f"More results exist — call again with offset={(offset or 0) + SEARCH_PAGE_SIZE}."
    result = ListResult(items=dumped, returned=len(dumped), note=note)
    return bounded_dump(result)


@platform_errors
async def get_user_profile(
    username: Annotated[str, Field(description="The platform username, e.g. 'ultralytics'")],
) -> dict[str, Any]:
    """Get a user's or team's public profile.

    Read-only — spends nothing. Returns display name, account type, bio, company and
    follower count. Browse their public work with search_platform(author=<username>)
    or list_datasets/list_projects with username=<username>.
    """
    token = get_request_token()
    data = await platform_api.get(
        "/api/users",
        token=token,
        params={"username": username},
        resource_hint=f"User '{username}'",
    )
    profile = UserProfile.from_api(data.get("user") or {})
    return {"user": profile.model_dump(exclude_none=True)}


def register(mcp) -> None:
    mcp.tool(search_platform)
    mcp.tool(get_user_profile)
