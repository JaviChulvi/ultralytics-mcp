"""Dataset tools (US2, FR-003). All read-only."""

from __future__ import annotations

import asyncio
import statistics
from typing import Annotated, Any

from fastmcp.exceptions import ToolError
from pydantic import Field

from ..auth import get_request_token
from ..errors import PlatformError, platform_errors
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
from ..settings import settings

SPLIT_NAMES = ("train", "val", "test")
IMAGE_FIELDS = ("hash", "name", "split", "width", "height", "label_count")
IMAGE_STATS_NOTE = (
    "'overall' summarizes platform-computed whole-dataset histograms — approximate "
    "at bin resolution; its 'unlabeled_images' appears only when exactly derivable. "
    "Per split, 'images', 'unlabeled_images' and 'error_images' are exact; "
    "'labels_per_image' and 'dimensions' come from a sample of up to {n} images."
)


def _parse_dataset_ref(ref: str) -> tuple[str | None, str]:
    """Split 'owner/slug', 'owner/datasets/slug' or a platform URL into (username, slug-or-id)."""
    ref = ref.strip().removeprefix("https://").removeprefix("http://")
    ref = ref.removeprefix("platform.ultralytics.com").strip("/")
    parts = [p for p in ref.split("/") if p and p != "datasets"]
    if len(parts) >= 2:
        return parts[0], parts[-1]
    return None, parts[0] if parts else ref


def _match_exact(matches: list[DatasetSummary], ref: str) -> list[DatasetSummary]:
    """The platform's slug filter is fuzzy — narrow to exact slug, else exact name matches."""
    wanted = ref.lower()
    by_slug = [m for m in matches if (m.slug or "").lower() == wanted]
    return by_slug or [m for m in matches if m.name.lower() == wanted]


async def _resolve_dataset(
    token: str, dataset: str, username: str | None = None
) -> tuple[str | list[DatasetSummary], str | None, dict[str, Any] | None]:
    """Resolve a dataset reference to (id, owner, detail), or (candidates, owner, None).

    The detail endpoint accepts a slug directly, so a successful direct lookup also
    returns the detail payload for reuse. Falls back to list-and-match-exactly when
    the direct lookup fails — the contract says slugs work in the path, but the live
    API has been seen rejecting them with 400 as well as 404 (D11).
    """
    ref_username, ref = _parse_dataset_ref(dataset)
    username = username or ref_username
    if looks_like_object_id(ref):
        return ref, username, None
    owner_params = {"username": username} if username else None
    try:
        detail = await platform_api.get(
            f"/api/datasets/{ref}",
            token=token,
            params=owner_params,
            resource_hint=f"Dataset '{dataset}'",
        )
    except PlatformError as exc:
        if exc.status not in (400, 404):
            raise
    else:
        raw = detail.get("dataset") or {}
        if raw.get("_id"):
            return str(raw["_id"]), username, detail
    # The slug+username listing combination can return nothing for datasets that
    # exist, so when an owner is known fetch their list and match exactly here.
    params: dict[str, Any] = {"limit": settings.max_page_size}
    if username:
        params["username"] = username
    else:
        params["slug"] = ref
    data = await platform_api.get(
        "/api/datasets", token=token, params=params, resource_hint=f"Dataset '{dataset}'"
    )
    matches = _match_exact([DatasetSummary.from_api(i) for i in data.get("datasets", [])], ref)
    if not matches:
        raise ToolError(
            f"Dataset '{dataset}' was not found. Use list_datasets to see what's available; "
            "another user's public dataset can be referenced as 'username/slug'."
        )
    if len(matches) > 1:
        return matches, username, None
    return matches[0].id, username, None


def _candidates(matches: list[DatasetSummary], ref: str) -> dict[str, Any]:
    return {
        "candidates": [m.model_dump(exclude_none=True) for m in matches],
        "note": f"Multiple datasets match '{ref}' — call again with the exact id.",
    }


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
    total = data.get("total")
    if slug:
        # The platform's slug filter is fuzzy and unreliable next to username: match
        # exactly here, refetching without it when nothing matched.
        matched = _match_exact(datasets, slug)
        if not matched:
            wide = {k: v for k, v in params.items() if k != "slug"}
            wide["limit"] = settings.max_page_size
            data = await platform_api.get(
                "/api/datasets", token=token, params=wide, resource_hint="Your dataset list"
            )
            matched = _match_exact(
                [DatasetSummary.from_api(i) for i in data.get("datasets", [])], slug
            )
        datasets = matched
        total = len(matched)
    empty_note = (
        "No datasets matched these filters — check the slug/username or drop a filter."
        if slug or username
        else "No datasets yet — create one at platform.ultralytics.com."
    )
    return make_list_result(datasets, total=total, empty_note=empty_note)


def _histogram_summary(bins: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Collapse a platform histogram ({bin, count, size?} entries) to min/median/max."""
    occupied = sorted((b for b in bins if b.get("count")), key=lambda b: b["bin"])
    if not occupied:
        return None
    total = sum(b["count"] for b in occupied)
    median = occupied[-1]["bin"]
    acc = 0
    for b in occupied:
        acc += b["count"]
        if acc * 2 >= total:
            median = b["bin"]
            break
    last = occupied[-1]
    return {
        "min": occupied[0]["bin"],
        "median": median,
        "max": last["bin"] + max(int(last.get("size", 1)) - 1, 0),
    }


def _overall_image_stats(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Whole-dataset distributions the platform precomputes alongside class stats."""
    out: dict[str, Any] = {}
    objects = raw.get("objectsPerImageHistogram") or []
    labels = _histogram_summary(objects)
    if labels:
        out["labels_per_image"] = labels
        # A ranged zero-bin lumps 0-label images with lightly-labeled ones; only
        # report the count when it is exactly derivable.
        zero_bins = [b for b in objects if b.get("bin") == 0 and b.get("count")]
        if not zero_bins:
            out["unlabeled_images"] = 0
        elif all(b.get("size", 1) == 1 for b in zero_bins):
            out["unlabeled_images"] = sum(b["count"] for b in zero_bins)
    for key, hist in (("width", raw.get("widthHistogram")), ("height", raw.get("heightHistogram"))):
        summary = _histogram_summary(hist or [])
        if summary:
            out[key] = {"min": summary["min"], "max": summary["max"]}
    if raw.get("formatDistribution"):
        out["formats"] = raw["formatDistribution"]
    return out or None


async def _split_image_stats(
    token: str, dataset_id: str, split: str, username: str | None, hint: str
) -> tuple[str, dict[str, Any]]:
    params: dict[str, Any] = {"split": split, "includeTotal": True, "includeThumbnails": False}
    if username:
        params["username"] = username
    path = f"/api/datasets/{dataset_id}/images"
    sample_params = {**params, "limit": settings.stats_sample_limit}
    unlabeled_params = {**params, "limit": 1, "hasLabel": False}
    page, unlabeled = await asyncio.gather(
        platform_api.get(path, token=token, params=sample_params, resource_hint=hint),
        platform_api.get(path, token=token, params=unlabeled_params, resource_hint=hint),
    )
    images = [DatasetImage.from_api(item) for item in page.get("images", [])]
    stats: dict[str, Any] = {
        "images": page.get("total"),
        "unlabeled_images": unlabeled.get("total"),
    }
    if page.get("errorCount"):
        stats["error_images"] = page["errorCount"]
    counts = sorted(img.label_count for img in images if img.label_count is not None)
    if counts:
        stats["labels_per_image"] = {
            "min": counts[0],
            "median": statistics.median(counts),
            "max": counts[-1],
            "sample_size": len(counts),
        }
    dims = {(img.width, img.height) for img in images if img.width and img.height}
    if dims:
        smallest = min(dims, key=lambda d: d[0] * d[1])
        largest = max(dims, key=lambda d: d[0] * d[1])
        stats["dimensions"] = {
            "smallest": f"{smallest[0]}x{smallest[1]}",
            "largest": f"{largest[0]}x{largest[1]}",
            "distinct_in_sample": len(dims),
        }
    return split, stats


async def _image_stats(
    token: str, dataset_id: str, summary: DatasetSummary, username: str | None, hint: str
) -> dict[str, Any]:
    if summary.splits:
        splits = [s for s in SPLIT_NAMES if (summary.splits.get(s) or 0) > 0]
    else:
        splits = list(SPLIT_NAMES)
    results = await asyncio.gather(
        *(_split_image_stats(token, dataset_id, s, username, hint) for s in splits)
    )
    return dict(results)


@platform_errors
async def get_dataset(
    dataset: Annotated[
        str,
        Field(
            description="Dataset id (24-char hex), slug, or 'username/slug' for another "
            "user's public dataset"
        ),
    ],
    username: Annotated[
        str | None,
        Field(description="Owner username when referencing another user's public dataset"),
    ] = None,
    include_image_stats: Annotated[
        bool,
        Field(
            description="Also compute per-split image statistics: exact unlabeled and "
            "error image counts plus sampled labels-per-image and dimension ranges"
        ),
    ] = False,
) -> dict[str, Any]:
    """Get one dataset's details including per-class and whole-dataset image statistics.

    Read-only — spends nothing. Returns name, task, image count, split sizes, class
    names, per-class instance/image counts and exact whole-dataset distributions
    (labels per image, unlabeled count, dimensions); set include_image_stats to add
    per-split breakdowns (saves paging through images). If a slug matches several
    datasets, the candidates are returned instead of guessing.
    """
    token = get_request_token()
    resolved, username, detail = await _resolve_dataset(token, dataset, username)
    if isinstance(resolved, list):
        return _candidates(resolved, dataset)
    hint = f"Dataset '{dataset}'"
    stats_call = platform_api.get(
        f"/api/datasets/{resolved}/class-stats", token=token, resource_hint=hint
    )
    if detail is None:
        owner_params = {"username": username} if username else None
        detail, stats = await asyncio.gather(
            platform_api.get(
                f"/api/datasets/{resolved}", token=token, params=owner_params, resource_hint=hint
            ),
            stats_call,
        )
    else:
        stats = await stats_call
    raw = detail.get("dataset", {})
    summary = DatasetSummary.from_api(raw)
    # Name classes from the untruncated upstream list so every class id resolves.
    class_names = raw.get("classNames") or stats.get("classNames") or []
    classes = [ClassStat.from_api(item, class_names) for item in stats.get("classes", [])]
    image_stats: dict[str, Any] = {}
    overall = _overall_image_stats(stats.get("imageStats") or {})
    if overall:
        image_stats["overall"] = overall
    if include_image_stats:
        image_stats["splits"] = await _image_stats(token, resolved, summary, username, hint)
    result = DatasetDetail(
        dataset=summary,
        classes=classes or None,
        stats_sampled=stats.get("sampled"),
        stats_sample_size=stats.get("sampleSize"),
        image_stats=image_stats or None,
        image_stats_note=(
            IMAGE_STATS_NOTE.format(n=settings.stats_sample_limit) if image_stats else None
        ),
    )
    return bounded_dump(result)


@platform_errors
async def list_dataset_images(
    dataset: Annotated[
        str,
        Field(
            description="Dataset id (24-char hex), slug, or 'username/slug' for another "
            "user's public dataset"
        ),
    ],
    split: Annotated[str | None, Field(description="Filter by split: train, val or test")] = None,
    limit: Annotated[
        int | None, Field(description="Max images to return (default 20, max 50)", ge=1)
    ] = None,
    offset: Annotated[int | None, Field(description="Skip this many images", ge=0)] = None,
    cursor: Annotated[
        str | None, Field(description="Continuation cursor from a previous page")
    ] = None,
    has_label: Annotated[
        bool | None,
        Field(description="Only images with labels (true) or without labels (false)"),
    ] = None,
    fields: Annotated[
        list[str] | None,
        Field(
            description="Per-image fields to return, e.g. ['name', 'split'] — omit for "
            "all of: hash, name, split, width, height, label_count"
        ),
    ] = None,
    username: Annotated[
        str | None,
        Field(description="Owner username when referencing another user's public dataset"),
    ] = None,
) -> dict[str, Any]:
    """List images in a dataset, paged.

    Read-only — spends nothing. Returns per-image hash, name, split, dimensions and
    label count (trim with fields to fit more per page), plus the total and a
    continuation cursor for the next page.
    """
    token = get_request_token()
    if fields:
        unknown = sorted(set(fields) - set(IMAGE_FIELDS))
        if unknown:
            raise ToolError(
                f"Unknown image fields {unknown} — choose from: {', '.join(IMAGE_FIELDS)}."
            )
    resolved, username, _ = await _resolve_dataset(token, dataset, username)
    if isinstance(resolved, list):
        return _candidates(resolved, dataset)
    params: dict[str, Any] = {
        "limit": clamp_limit(limit),
        "includeTotal": True,
        # Upstream includes signed thumbnail URLs by default; we never surface them.
        "includeThumbnails": False,
    }
    if split:
        params["split"] = split
    if offset is not None:
        params["offset"] = offset
    if cursor:
        params["cursor"] = cursor
    if has_label is not None:
        params["hasLabel"] = has_label
    if username:
        params["username"] = username
    data = await platform_api.get(
        f"/api/datasets/{resolved}/images",
        token=token,
        params=params,
        resource_hint=f"Images of dataset '{dataset}'",
    )
    images = [DatasetImage.from_api(item) for item in data.get("images", [])]
    items = [img.model_dump(exclude_none=True) for img in images]
    if fields:
        keep = set(fields)
        items = [{k: v for k, v in item.items() if k in keep} for item in items]
    page = DatasetImagePage(
        items=items,
        returned=len(items),
        total=data.get("total"),
        has_more=data.get("hasMore"),
        next_cursor=data.get("nextCursor"),
        note="No images in this dataset (or split) yet." if not items else None,
    )
    return bounded_dump(page)


def register(mcp) -> None:
    mcp.tool(list_datasets)
    mcp.tool(get_dataset)
    mcp.tool(list_dataset_images)
