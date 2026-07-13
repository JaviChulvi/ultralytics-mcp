"""Output models: field whitelists over raw platform payloads (FR-006, SC-005).

Each model's ``from_api`` picks only decision-relevant fields from the upstream
response, so new upstream fields can never silently bloat tool output. Field names
follow tests/fixtures/openapi.json (the vendored upstream contract).
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from .settings import settings

TRUNCATION_NOTE = (
    "Response truncated to stay compact — use filters (slug/username/status) "
    "or pagination to narrow it down."
)

# Long class lists (e.g. 80 COCO names) dominate list payloads; previews keep them
# readable while class_names_omitted says what was cut. get_dataset keeps full stats.
CLASS_NAMES_PREVIEW = 15


def clamp_limit(limit: int | None) -> int:
    """Apply the default page size and the hard cap (SC-005)."""
    if limit is None:
        return settings.default_page_size
    return max(1, min(limit, settings.max_page_size))


def looks_like_object_id(ref: str) -> bool:
    """Platform ids are 24-char hex; anything else is treated as a slug (D11)."""
    return len(ref) == 24 and all(c in "0123456789abcdef" for c in ref.lower())


class ProjectSummary(BaseModel):
    id: str
    name: str
    slug: str | None = None
    visibility: str | None = None
    model_count: int | None = None
    updated_at: str | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> ProjectSummary:
        return cls(
            id=str(data.get("_id", "")),
            name=data.get("name", ""),
            slug=data.get("slug"),
            visibility=data.get("visibility"),
            model_count=data.get("modelCount"),
            updated_at=data.get("updatedAt"),
        )


class DatasetSummary(BaseModel):
    id: str
    name: str
    slug: str | None = None
    task: str | None = None
    visibility: str | None = None
    image_count: int | None = None
    class_count: int | None = None
    class_names: list[str] | None = None
    class_names_omitted: int | None = None
    splits: dict[str, Any] | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> DatasetSummary:
        class_names = data.get("classNames")
        omitted = None
        if class_names and len(class_names) > CLASS_NAMES_PREVIEW:
            omitted = len(class_names) - CLASS_NAMES_PREVIEW
            class_names = class_names[:CLASS_NAMES_PREVIEW]
        return cls(
            id=str(data.get("_id", "")),
            name=data.get("name", ""),
            slug=data.get("slug"),
            task=data.get("task"),
            visibility=data.get("visibility"),
            image_count=data.get("imageCount"),
            class_count=data.get("classCount"),
            class_names=class_names,
            class_names_omitted=omitted,
            splits=data.get("splits"),
        )


class ClassStat(BaseModel):
    class_id: int | None = None
    name: str | None = None
    instance_count: int | None = None
    image_count: int | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any], class_names: list[str] | None = None) -> ClassStat:
        class_id = data.get("classId")
        name = None
        if class_names and isinstance(class_id, int) and 0 <= class_id < len(class_names):
            name = class_names[class_id]
        return cls(
            class_id=class_id,
            name=name,
            instance_count=data.get("count"),
            image_count=data.get("imageCount"),
        )


class DatasetDetail(BaseModel):
    dataset: DatasetSummary
    classes: list[ClassStat] | None = None
    stats_sampled: bool | None = None
    stats_sample_size: int | None = None
    image_stats: dict[str, Any] | None = None
    image_stats_note: str | None = None


class DatasetImage(BaseModel):
    hash: str | None = None
    name: str | None = None
    split: str | None = None
    width: int | None = None
    height: int | None = None
    label_count: int | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> DatasetImage:
        return cls(
            hash=data.get("hash"),
            name=data.get("name"),
            split=data.get("split"),
            width=data.get("width"),
            height=data.get("height"),
            label_count=data.get("labelCount"),
        )


class ModelSummary(BaseModel):
    id: str
    name: str
    slug: str | None = None
    project_id: str | None = None
    task: str | None = None
    status: str | None = None
    epochs: int | None = None
    best_epoch: int | None = None
    best_fitness: float | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> ModelSummary:
        return cls(
            id=str(data.get("_id", "")),
            name=data.get("name", ""),
            slug=data.get("slug"),
            project_id=data.get("projectId"),
            task=data.get("task"),
            status=data.get("status"),
            epochs=data.get("epochs"),
            best_epoch=data.get("bestEpoch"),
            best_fitness=data.get("bestFitness"),
        )


class TrainingStatus(BaseModel):
    model_id: str
    status: str | None = None
    epochs_total: int | None = None
    epochs_completed: int | None = None
    progress_percent: float | None = None
    latest_metrics: dict[str, Any] | None = None
    note: str | None = None

    @classmethod
    def from_api(cls, model_id: str, data: dict[str, Any]) -> TrainingStatus:
        results = data.get("trainResults") or []
        total = data.get("epochs")
        completed = len(results)
        progress = round(100 * completed / total, 1) if total else None
        return cls(
            model_id=model_id,
            status=data.get("status"),
            epochs_total=total,
            epochs_completed=completed or None,
            progress_percent=progress,
            latest_metrics=results[-1] if results else None,
        )


class ExportSummary(BaseModel):
    id: str
    model_id: str | None = None
    format: str | None = None
    status: str | None = None
    file: str | None = None
    error: str | None = None
    completed_at: str | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> ExportSummary:
        return cls(
            id=str(data.get("_id", "")),
            model_id=data.get("modelId"),
            format=data.get("format"),
            status=data.get("status"),
            file=data.get("file"),
            error=data.get("error"),
            completed_at=data.get("completedAt"),
        )


class DeploymentSummary(BaseModel):
    id: str
    name: str
    model_id: str | None = None
    status: str | None = None
    status_message: str | None = None
    region: str | None = None
    service_url: str | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> DeploymentSummary:
        return cls(
            id=str(data.get("_id", "")),
            name=data.get("name", ""),
            model_id=data.get("modelId"),
            status=data.get("status"),
            status_message=data.get("statusMessage"),
            region=data.get("region"),
            service_url=data.get("serviceUrl"),
        )


class DeploymentStatus(BaseModel):
    deployment: DeploymentSummary
    healthy: bool | None = None
    health_error: str | None = None
    latency_ms: float | None = None
    total_requests: int | None = None
    error_rate: float | None = None
    avg_latency_ms: float | None = None


class AccountStatus(BaseModel):
    credits_cents: float | None = None
    plan: str | None = None
    storage_tier: str | None = None
    storage_used_bytes: float | None = None


class ActivityItem(BaseModel):
    action: str | None = None
    resource_type: str | None = None
    resource_name: str | None = None
    timestamp: str | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> ActivityItem:
        return cls(
            action=data.get("action"),
            resource_type=data.get("resourceType"),
            resource_name=data.get("resourceName"),
            timestamp=data.get("timestamp"),
        )


class DatasetImagePage(BaseModel):
    items: list[dict[str, Any]]
    returned: int
    total: int | None = None
    has_more: bool | None = None
    next_cursor: str | None = None
    truncated: bool = False
    note: str | None = None


class ListResult(BaseModel):
    """Uniform envelope for list tools: bounded, with an explicit truncation signal."""

    items: list[dict[str, Any]]
    returned: int
    total: int | None = None
    truncated: bool = False
    note: str | None = None


def make_list_result(
    models: list[BaseModel], total: int | None = None, empty_note: str | None = None
) -> dict[str, Any]:
    items = [m.model_dump(exclude_none=True, mode="json") for m in models]
    truncated = total is not None and total > len(items)
    result = ListResult(
        items=items,
        returned=len(items),
        total=total,
        truncated=truncated,
        note=(empty_note if not items else (TRUNCATION_NOTE if truncated else None)),
    )
    return bounded_dump(result)


def bounded_dump(model: BaseModel) -> dict[str, Any]:
    """Serialize a model, trimming list items if needed to respect the size cap (SC-005)."""
    data = model.model_dump(exclude_none=True, mode="json")
    if len(json.dumps(data)) <= settings.max_response_bytes:
        return data
    items = data.get("items")
    if isinstance(items, list):
        while items and len(json.dumps(data)) > settings.max_response_bytes:
            items.pop()
        data["returned"] = len(items)
        data["truncated"] = True
        data["note"] = TRUNCATION_NOTE
    return data
