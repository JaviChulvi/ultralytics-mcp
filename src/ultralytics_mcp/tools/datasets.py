"""Dataset tools (US2, FR-003): browse, download, import and edit datasets.

Read-only tools and state-changing tools live side by side; every docstring says
which it is. Nothing here spends credits — dataset work costs storage quota only.
"""

from __future__ import annotations

import asyncio
import re
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
    ListResult,
    ModelSummary,
    bounded_dump,
    clamp_limit,
    looks_like_object_id,
    make_list_result,
)
from ..settings import settings

SPLIT_NAMES = ("train", "val", "test")
TASK_TYPES = ("detect", "segment", "semantic", "classify", "pose", "obb")
VISIBILITIES = ("public", "private")
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


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not slug:
        raise ToolError("The name must contain at least one letter or number.")
    return slug


async def _resolve_dataset_id_or_candidates(
    token: str, dataset: str, username: str | None
) -> tuple[str, str | None] | dict[str, Any]:
    """Resolve to (id, owner) for action tools, or the candidates payload if ambiguous."""
    resolved, username, _ = await _resolve_dataset(token, dataset, username)
    if isinstance(resolved, list):
        return _candidates(resolved, dataset)
    return resolved, username


@platform_errors
async def get_dataset_download(
    dataset: Annotated[
        str, Field(description="Dataset id (24-char hex), slug, or 'username/slug'")
    ],
    version: Annotated[
        int | None,
        Field(description="A saved version number — omit for the current data", ge=1),
    ] = None,
    username: Annotated[
        str | None,
        Field(description="Owner username when referencing another user's public dataset"),
    ] = None,
) -> dict[str, Any]:
    """Get a download link for a dataset (current data or a saved version).

    Read-only — spends nothing. Returns a signed URL to an NDJSON export: one
    dataset-metadata line, then one line per image with a signed image URL and its
    annotations. The link stays valid for about 7 days. Fails with a conflict while
    the dataset is still processing an import.
    """
    token = get_request_token()
    resolved = await _resolve_dataset_id_or_candidates(token, dataset, username)
    if isinstance(resolved, dict):
        return resolved
    dataset_id, _ = resolved
    params = {"v": version} if version is not None else None
    data = await platform_api.get(
        f"/api/datasets/{dataset_id}/export",
        token=token,
        params=params,
        resource_hint=f"Download of dataset '{dataset}'",
    )
    result: dict[str, Any] = {
        "download_url": data.get("downloadUrl"),
        "format": "ndjson",
        "note": "Signed URL, valid ~7 days. NDJSON: first line is dataset metadata, "
        "then one line per image with a signed image URL and annotations.",
    }
    if data.get("version") is not None:
        result["version"] = data["version"]
    return result


@platform_errors
async def create_dataset_version(
    dataset: Annotated[
        str, Field(description="Dataset id (24-char hex), slug, or 'username/slug'")
    ],
    description: Annotated[
        str | None, Field(description="What this snapshot captures, e.g. 'before relabeling'")
    ] = None,
) -> dict[str, Any]:
    """Snapshot a dataset as an immutable numbered version.

    State-changing — spends no credits. Captures the dataset's current images and
    labels so they can be downloaded (get_dataset_download) or restored later even
    after edits. Take one before bulk label changes.
    """
    token = get_request_token()
    resolved = await _resolve_dataset_id_or_candidates(token, dataset, None)
    if isinstance(resolved, dict):
        return resolved
    dataset_id, _ = resolved
    body: dict[str, Any] = {"description": description} if description else {}
    data = await platform_api.post(
        f"/api/datasets/{dataset_id}/export",
        token=token,
        json=body,
        resource_hint=f"Version snapshot of dataset '{dataset}'",
    )
    return {
        "version": data.get("version"),
        "download_url": data.get("downloadUrl"),
        "note": "Immutable snapshot created — restore or download it by this version number.",
    }


@platform_errors
async def list_dataset_models(
    dataset: Annotated[
        str, Field(description="Dataset id (24-char hex), slug, or 'username/slug'")
    ],
    username: Annotated[
        str | None,
        Field(description="Owner username when referencing another user's public dataset"),
    ] = None,
) -> dict[str, Any]:
    """List the models that were trained on a dataset (its lineage).

    Read-only — spends nothing. Returns each model's id, name, project, training
    status and best fitness — useful to see whether a dataset already has trained
    models before starting a new run.
    """
    token = get_request_token()
    resolved = await _resolve_dataset_id_or_candidates(token, dataset, username)
    if isinstance(resolved, dict):
        return resolved
    dataset_id, owner = resolved
    params = {"username": owner} if owner else None
    data = await platform_api.get(
        f"/api/datasets/{dataset_id}/models",
        token=token,
        params=params,
        resource_hint=f"Models trained on dataset '{dataset}'",
    )
    items = []
    for raw in data.get("models", []):
        item = ModelSummary.from_api(raw).model_dump(exclude_none=True)
        if raw.get("projectSlug"):
            item["project_slug"] = raw["projectSlug"]
        if raw.get("username"):
            item["username"] = raw["username"]
        items.append(item)
    result = ListResult(
        items=items,
        returned=len(items),
        total=data.get("count"),
        note="No models have been trained on this dataset yet." if not items else None,
    )
    return bounded_dump(result)


async def _create_dataset(token: str, name: str, task: str, **extra: Any) -> dict[str, Any]:
    if task not in TASK_TYPES:
        raise ToolError(f"Unknown task '{task}' — choose from: {', '.join(TASK_TYPES)}.")
    body: dict[str, Any] = {"name": name, "slug": _slugify(name), "task": task}
    body.update({k: v for k, v in extra.items() if v})
    data = await platform_api.post(
        "/api/datasets", token=token, json=body, resource_hint=f"Dataset '{name}'"
    )
    return {"dataset_id": data.get("datasetId"), "slug": data.get("slug")}


@platform_errors
async def create_dataset(
    name: Annotated[str, Field(description="Human-readable dataset name")],
    task: Annotated[
        str, Field(description="YOLO task: detect, segment, semantic, classify, pose or obb")
    ] = "detect",
    description: Annotated[str | None, Field(description="What the dataset contains")] = None,
    visibility: Annotated[
        str | None, Field(description="'public' or 'private' (platform default: private)")
    ] = None,
    class_names: Annotated[
        list[str] | None, Field(description="Initial class names, in index order")
    ] = None,
    owner: Annotated[
        str | None, Field(description="Team username, to create in a team workspace")
    ] = None,
) -> dict[str, Any]:
    """Create an empty dataset record.

    State-changing — spends no credits. Makes the record only; add images with
    import_dataset_from_url (or the platform UI for local files). The slug is
    derived from the name and de-duplicated by the platform.
    """
    if visibility is not None and visibility not in VISIBILITIES:
        raise ToolError(f"visibility must be one of: {', '.join(VISIBILITIES)}.")
    token = get_request_token()
    created = await _create_dataset(
        token,
        name,
        task,
        description=description,
        visibility=visibility,
        classNames=class_names,
        owner=owner,
    )
    return {
        **created,
        "note": "Empty dataset created — import images with import_dataset_from_url.",
    }


@platform_errors
async def update_dataset(
    dataset: Annotated[str, Field(description="Dataset id (24-char hex) or slug")],
    name: Annotated[
        str | None, Field(description="New name (the slug changes with it)")
    ] = None,
    description: Annotated[str | None, Field(description="New description")] = None,
    visibility: Annotated[str | None, Field(description="'public' or 'private'")] = None,
    tags: Annotated[list[str] | None, Field(description="Replacement tag list")] = None,
) -> dict[str, Any]:
    """Rename or edit a dataset's metadata.

    State-changing — spends no credits. Updates only the fields you pass. Renaming
    changes the slug (the response returns the new one) and keeps dependent models'
    references intact.
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
    if tags is not None:
        updates["tags"] = tags
    if not updates:
        raise ToolError("Nothing to update — pass at least one of name/description/visibility/tags.")
    token = get_request_token()
    resolved = await _resolve_dataset_id_or_candidates(token, dataset, None)
    if isinstance(resolved, dict):
        return resolved
    dataset_id, _ = resolved
    data = await platform_api.patch(
        f"/api/datasets/{dataset_id}",
        token=token,
        json=updates,
        resource_hint=f"Dataset '{dataset}'",
    )
    return {"success": bool(data.get("success")), "slug": data.get("slug")}


@platform_errors
async def delete_dataset(
    dataset: Annotated[str, Field(description="Dataset id (24-char hex) or slug")],
) -> dict[str, Any]:
    """Move a dataset to the trash (soft delete).

    State-changing — spends no credits. The dataset is recoverable for 30 days with
    restore_from_trash; after that a daily cleanup removes it permanently. Trashed
    items still count toward storage.
    """
    token = get_request_token()
    resolved = await _resolve_dataset_id_or_candidates(token, dataset, None)
    if isinstance(resolved, dict):
        return resolved
    dataset_id, _ = resolved
    await platform_api.delete(
        f"/api/datasets/{dataset_id}", token=token, resource_hint=f"Dataset '{dataset}'"
    )
    return {
        "success": True,
        "dataset_id": dataset_id,
        "note": "Moved to trash — recoverable for 30 days with restore_from_trash.",
    }


@platform_errors
async def import_dataset_from_url(
    source_url: Annotated[
        str,
        Field(
            description="Public or signed URL of a dataset archive (.zip, .tar, .tar.gz, "
            ".tgz) or .ndjson file in YOLO/COCO/VOC layout"
        ),
    ],
    dataset: Annotated[
        str | None,
        Field(description="Existing dataset (id or slug) to import into — omit to create one"),
    ] = None,
    name: Annotated[
        str | None,
        Field(description="Name for a NEW dataset — provide this or 'dataset', not both"),
    ] = None,
    task: Annotated[
        str, Field(description="Task for a new dataset: detect, segment, classify, pose, obb")
    ] = "detect",
    target_split: Annotated[
        str | None,
        Field(description="Force all imported images into one split: train, val or test"),
    ] = None,
) -> dict[str, Any]:
    """Import images and labels into a dataset from an archive URL.

    State-changing — spends no credits (storage quota applies). Creates the dataset
    when you pass a name, then queues an asynchronous ingest job that downloads,
    extracts and validates the archive. Poll get_dataset until its status is 'ready';
    a conflict error means an import is already running.
    """
    if (dataset is None) == (name is None):
        raise ToolError("Provide exactly one of 'dataset' (existing) or 'name' (create new).")
    if target_split is not None and target_split not in SPLIT_NAMES:
        raise ToolError(f"target_split must be one of: {', '.join(SPLIT_NAMES)}.")
    token = get_request_token()
    slug = None
    if name is not None:
        created = await _create_dataset(token, name, task)
        dataset_id, slug = created["dataset_id"], created["slug"]
    else:
        resolved = await _resolve_dataset_id_or_candidates(token, dataset, None)
        if isinstance(resolved, dict):
            return resolved
        dataset_id, _ = resolved
    body: dict[str, Any] = {"datasetId": dataset_id, "sourceUrl": source_url}
    if target_split:
        body["targetSplit"] = target_split
    data = await platform_api.post(
        "/api/datasets/ingest",
        token=token,
        json=body,
        resource_hint=f"Import into dataset '{name or dataset}'",
    )
    result: dict[str, Any] = {
        "dataset_id": dataset_id,
        "job_id": data.get("jobId"),
        "status": data.get("status", "queued"),
        "note": "Import runs asynchronously — poll get_dataset until the dataset "
        "status is 'ready' (large archives can take a while).",
    }
    if slug:
        result["slug"] = slug
    return result


def register(mcp) -> None:
    mcp.tool(list_datasets)
    mcp.tool(get_dataset)
    mcp.tool(list_dataset_images)
    mcp.tool(get_dataset_download)
    mcp.tool(create_dataset_version)
    mcp.tool(list_dataset_models)
    mcp.tool(create_dataset)
    mcp.tool(update_dataset)
    mcp.tool(delete_dataset)
    mcp.tool(import_dataset_from_url)
