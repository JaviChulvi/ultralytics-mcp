"""Account and public discovery tools."""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from ..errors import sdk_errors
from ..runtime import runtime
from .common import provided, query_bool


@sdk_errors
async def get_account_status() -> dict[str, Any]:
    """Get the authenticated workspace, plan, credits, storage, and resource counts.

    Read-only and spends nothing. This is also the best first call to verify that
    ULTRALYTICS_API_KEY is valid.
    """
    client = runtime.sdk()
    summary, storage = await asyncio.gather(client.account.summary(), client.account.storage())
    runtime._default_owner = summary["username"]
    return {"summary": summary, "storage": storage}


@sdk_errors
async def search_platform(
    q: str | None = None,
    type: Literal["all", "projects", "datasets"] | None = None,
    sort: Literal["stars", "newest", "oldest", "name-asc", "name-desc", "count-desc", "count-asc"]
    | None = None,
    offset: int | None = None,
    limit: int = 20,
    task: str | None = None,
    author: str | None = None,
    starred: bool | None = None,
) -> dict[str, Any]:
    """Search public Platform projects and datasets.

    Read-only and spends nothing. Results identify their owners and URL slugs for use
    with the other tools.
    """
    return await runtime.sdk().explore.search(
        **provided(
            q=q,
            type=type,
            sort=sort,
            offset=offset,
            limit=max(1, min(limit, 50)),
            task=task,
            author=author,
            starred=query_bool(starred),
        )
    )


def register(mcp) -> None:
    mcp.tool(get_account_status)
    mcp.tool(search_platform)
