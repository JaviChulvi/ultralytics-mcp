"""Dataset tools backed directly by the official SDK."""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from ..errors import sdk_errors
from ..runtime import runtime
from .common import provided, query_bool

Task = Literal["detect", "segment", "semantic", "depth", "classify", "pose", "obb"]


@sdk_errors
async def list_datasets(
    owner: str | None = None,
    limit: int = 20,
    include_samples: bool = False,
    include_image_urls: bool = False,
) -> dict[str, Any]:
    """List datasets in a personal or team workspace. Read-only and spends nothing."""
    return await runtime.sdk().datasets.list(
        await runtime.owner(owner),
        limit=max(1, min(limit, 50)),
        include_samples=query_bool(include_samples),
        include_image_urls=query_bool(include_image_urls),
    )


@sdk_errors
async def get_dataset(
    dataset: str, owner: str | None = None, include_stats: bool = True
) -> dict[str, Any]:
    """Get a dataset by owner and slug, optionally including class and image statistics.

    Read-only and spends nothing. With statistics enabled, the raw SDK responses are
    returned under `dataset` and `classStats`.
    """
    resolved_owner = await runtime.owner(owner)
    client = runtime.sdk()
    if not include_stats:
        return await client.datasets.retrieve(resolved_owner, dataset)
    detail, stats = await asyncio.gather(
        client.datasets.retrieve(resolved_owner, dataset),
        client.datasets.class_stats(resolved_owner, dataset),
    )
    return {"dataset": detail, "classStats": stats}


@sdk_errors
async def create_dataset(
    dataset: str,
    name: str,
    owner: str | None = None,
    task: Task | None = None,
    description: str | None = None,
    visibility: Literal["public", "private"] | None = None,
    class_names: list[str] | None = None,
    format: Literal["yolo", "coco", "raw", "ndjson"] | None = None,
    tags: list[str] | None = None,
    license: str | None = None,
) -> dict[str, Any]:
    """Create an empty dataset. State-changing but does not spend training credits."""
    return await runtime.sdk().datasets.create(
        **provided(
            dataset=dataset,
            name=name,
            owner=await runtime.owner(owner),
            task=task,
            description=description,
            visibility=visibility,
            class_names=class_names,
            format=format,
            tags=tags,
            license=license,
        )
    )


@sdk_errors
async def import_dataset_from_url(
    dataset: str,
    source_url: str,
    owner: str | None = None,
    target_split: Literal["train", "val", "test"] | None = None,
    conflict_policy: Literal["skip", "keep_both", "replace"] | None = None,
) -> dict[str, Any]:
    """Queue a remote archive or NDJSON URL for ingestion into an existing dataset.

    State-changing but spends no training credits. Poll get_dataset until processing
    reaches a terminal state.
    """
    body = provided(
        sourceUrl=source_url,
        targetSplit=target_split,
        conflictPolicy=conflict_policy,
    )
    return await runtime.sdk().datasets.ingest(await runtime.owner(owner), dataset, body=body)


@sdk_errors
async def get_dataset_download(
    dataset: str, owner: str | None = None, version: int | None = None
) -> dict[str, Any]:
    """Get a signed NDJSON download URL for a dataset or immutable version.

    Read-only and spends nothing.
    """
    return await runtime.sdk().datasets.export(
        await runtime.owner(owner), dataset, **provided(v=version)
    )


@sdk_errors
async def create_dataset_version(
    dataset: str, owner: str | None = None, description: str | None = None
) -> dict[str, Any]:
    """Create an immutable dataset version. State-changing but spends no credits."""
    return await runtime.sdk().datasets.create_export(
        await runtime.owner(owner), dataset, **provided(description=description)
    )


def register(mcp) -> None:
    for tool in (
        list_datasets,
        get_dataset,
        create_dataset,
        import_dataset_from_url,
        get_dataset_download,
        create_dataset_version,
    ):
        mcp.tool(tool)
