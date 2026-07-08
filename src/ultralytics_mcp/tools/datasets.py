"""Dataset tools (US2, FR-003). All read-only."""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

from fastmcp.exceptions import ToolError
from pydantic import Field

from ..auth import get_request_token
from ..errors import platform_errors
from ..platform_client import platform_api
from ..schemas import (
    ClassStat,
    DatasetDetail,
    DatasetImage,
    DatasetImagePage,
    DatasetSummary,
    bounded_dump,
    clamp_limit,
    looks_like_object_id,
    make_list_result,
)


@platform_errors
async def list_datasets(
    limit: Annotated[
        int | None, Field(description="Max datasets to return (default 20, max 50)", ge=1)
    ] = None,
    slug: Annotated[str | None, Field(description="Filter by exact dataset slug")] = None,
    username: Annotated[
        str | None,
        Field(description="Owner username — set it to browse another user's public datasets"),
    ] = None,
) -> dict[str, Any]:
    """List the datasets your Ultralytics Platform account can see.

    Read-only — spends nothing. Returns each dataset's id, name, task, image count,
    class count and split sizes, plus the account total. Narrow with slug/username;
    use get_dataset for class statistics.
    """
    token = get_request_token()
    params: dict[str, Any] = {"limit": clamp_limit(limit)}
    if slug:
        params["slug"] = slug
    if username:
        params["username"] = username
    data = await platform_api.get(
        "/api/datasets", token=token, params=params, resource_hint="Your dataset list"
    )
    datasets = [DatasetSummary.from_api(item) for item in data.get("datasets", [])]
    return make_list_result(
        datasets,
        total=data.get("total"),
        empty_note="No datasets yet — create one at platform.ultralytics.com.",
    )


async def _resolve_dataset_id(token: str, dataset: str) -> str | list[DatasetSummary]:
    """Return the dataset id, or the candidate list when a slug is ambiguous (D11)."""
    if looks_like_object_id(dataset):
        return dataset
    data = await platform_api.get(
        "/api/datasets",
        token=token,
        params={"slug": dataset, "limit": 5},
        resource_hint=f"Dataset '{dataset}'",
    )
    matches = [DatasetSummary.from_api(item) for item in data.get("datasets", [])]
    if not matches:
        raise ToolError(
            f"Dataset '{dataset}' was not found. Use list_datasets to see what's available."
        )
    if len(matches) > 1:
        return matches
    return matches[0].id


def _candidates(matches: list[DatasetSummary], ref: str) -> dict[str, Any]:
    return {
        "candidates": [m.model_dump(exclude_none=True) for m in matches],
        "note": f"Multiple datasets match '{ref}' — call again with the exact id.",
    }


@platform_errors
async def get_dataset(
    dataset: Annotated[str, Field(description="Dataset id (24-char hex) or slug")],
) -> dict[str, Any]:
    """Get one dataset's details including per-class statistics.

    Read-only — spends nothing. Returns name, task, image count, split sizes, class
    names and per-class instance/image counts. If a slug matches several datasets,
    the candidates are returned instead of guessing.
    """
    token = get_request_token()
    resolved = await _resolve_dataset_id(token, dataset)
    if isinstance(resolved, list):
        return _candidates(resolved, dataset)
    hint = f"Dataset '{dataset}'"
    detail, stats = await asyncio.gather(
        platform_api.get(f"/api/datasets/{resolved}", token=token, resource_hint=hint),
        platform_api.get(f"/api/datasets/{resolved}/class-stats", token=token, resource_hint=hint),
    )
    summary = DatasetSummary.from_api(detail.get("dataset", {}))
    class_names = summary.class_names or stats.get("classNames") or []
    classes = [ClassStat.from_api(item, class_names) for item in stats.get("classes", [])]
    result = DatasetDetail(
        dataset=summary, classes=classes or None, stats_sampled=stats.get("sampled")
    )
    return bounded_dump(result)


@platform_errors
async def list_dataset_images(
    dataset: Annotated[str, Field(description="Dataset id (24-char hex) or slug")],
    split: Annotated[str | None, Field(description="Filter by split: train, val or test")] = None,
    limit: Annotated[
        int | None, Field(description="Max images to return (default 20, max 50)", ge=1)
    ] = None,
    offset: Annotated[int | None, Field(description="Skip this many images", ge=0)] = None,
    cursor: Annotated[
        str | None, Field(description="Continuation cursor from a previous page")
    ] = None,
) -> dict[str, Any]:
    """List images in a dataset, paged.

    Read-only — spends nothing. Returns per-image hash, name, split, dimensions and
    label count, plus the total and a continuation cursor for the next page.
    """
    token = get_request_token()
    resolved = await _resolve_dataset_id(token, dataset)
    if isinstance(resolved, list):
        return _candidates(resolved, dataset)
    params: dict[str, Any] = {"limit": clamp_limit(limit), "includeTotal": True}
    if split:
        params["split"] = split
    if offset is not None:
        params["offset"] = offset
    if cursor:
        params["cursor"] = cursor
    data = await platform_api.get(
        f"/api/datasets/{resolved}/images",
        token=token,
        params=params,
        resource_hint=f"Images of dataset '{dataset}'",
    )
    images = [DatasetImage.from_api(item) for item in data.get("images", [])]
    page = DatasetImagePage(
        items=[img.model_dump(exclude_none=True) for img in images],
        returned=len(images),
        total=data.get("total"),
        has_more=data.get("hasMore"),
        next_cursor=data.get("nextCursor"),
        note="No images in this dataset (or split) yet." if not images else None,
    )
    return bounded_dump(page)


def register(mcp) -> None:
    mcp.tool(list_datasets)
    mcp.tool(get_dataset)
    mcp.tool(list_dataset_images)
