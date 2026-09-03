"""Model and cloud-training tools backed directly by the official SDK."""

from __future__ import annotations

from typing import Any, Literal

from fastmcp.exceptions import ToolError
from ultralytics_platform import APIConnectionError, APIError

from ..errors import sdk_errors
from ..runtime import runtime
from .common import provided

Task = Literal["detect", "segment", "semantic", "depth", "classify", "pose", "obb"]
RESERVED_TRAIN_ARGS = {"model", "data", "epochs", "imgsz", "batch"}


@sdk_errors
async def list_models(project: str, owner: str | None = None, limit: int = 20) -> dict[str, Any]:
    """List models in a project. Read-only and spends nothing."""
    return await runtime.sdk().models.list(
        await runtime.owner(owner), project, limit=max(1, min(limit, 50))
    )


@sdk_errors
async def get_model(
    project: str,
    model: str,
    owner: str | None = None,
    include_analysis: bool = False,
) -> dict[str, Any]:
    """Get a model by owner, project, and model slug. Read-only and spends nothing."""
    options = {"analysis": "1"} if include_analysis else {}
    return await runtime.sdk().models.retrieve(
        await runtime.owner(owner), project, model, **options
    )


@sdk_errors
async def get_model_files(project: str, model: str, owner: str | None = None) -> dict[str, Any]:
    """Get signed download links for a model's files. Read-only and spends nothing."""
    return await runtime.sdk().models.files(await runtime.owner(owner), project, model)


@sdk_errors
async def get_training_status(project: str, model: str, owner: str | None = None) -> dict[str, Any]:
    """Get live training progress and metrics. Read-only and spends nothing.

    Poll this tool until status becomes completed, failed, or cancelled.
    """
    return await runtime.sdk().models.training(await runtime.owner(owner), project, model)


@sdk_errors
async def get_gpu_availability() -> dict[str, Any]:
    """Get current managed cloud GPU availability. Read-only and spends nothing."""
    return await runtime.sdk().training.gpu_availability(managed="true")


@sdk_errors
async def start_training(
    project: str,
    dataset: str,
    model: str,
    name: str,
    gpu_type: str,
    owner: str | None = None,
    dataset_owner: str | None = None,
    task: Task | None = None,
    base_model: str = "yolo26n.pt",
    epochs: int = 100,
    imgsz: int | None = None,
    batch: int | None = None,
    extra_args: dict[str, Any] | None = None,
    capture_dataset_version: bool = True,
    confirm_spend: bool = False,
) -> dict[str, Any]:
    """Create a model and start paid cloud training.

    This is the only tool that spends credits. Before setting confirm_spend=true,
    inspect the dataset, account balance, and GPU availability and obtain the user's
    explicit approval. Poll get_training_status after a successful start.
    """
    if not confirm_spend:
        return {
            "started": False,
            "confirmation_required": True,
            "message": (
                "Training spends credits. Review get_account_status, get_dataset, and "
                "get_gpu_availability; obtain explicit approval; then retry with "
                "confirm_spend=true."
            ),
        }
    if not 1 <= epochs <= 10_000:
        raise ToolError("epochs must be between 1 and 10000.")
    overlaps = RESERVED_TRAIN_ARGS.intersection(extra_args or {})
    if overlaps:
        raise ToolError(
            "extra_args cannot override reserved arguments: " + ", ".join(sorted(overlaps))
        )

    resolved_owner = await runtime.owner(owner)
    resolved_dataset_owner = dataset_owner or resolved_owner
    train_args: dict[str, Any] = {
        **(extra_args or {}),
        "model": base_model,
        "data": f"ul://{resolved_dataset_owner}/datasets/{dataset}",
        "epochs": epochs,
    }
    if imgsz is not None:
        train_args["imgsz"] = imgsz
    if batch is not None:
        train_args["batch"] = batch

    client = runtime.sdk()
    created = await client.models.create(
        body=provided(
            owner=resolved_owner,
            project=project,
            model=model,
            name=name,
            task=task,
            trainArgs=train_args,
        )
    )
    try:
        training = await client.training.start(
            model_id=created["id"],
            train_args=train_args,
            gpu_type=gpu_type,
            capture_dataset_version=capture_dataset_version,
        )
    except APIError:
        try:
            await client.models.delete(resolved_owner, project, model)
        except (APIError, APIConnectionError):
            pass
        raise
    return {"model": created, "training": training}


@sdk_errors
async def cancel_training(
    project: str, model: str, owner: str | None = None, confirm: bool = False
) -> dict[str, Any]:
    """Cancel active paid training.

    Cancellation is irreversible and elapsed GPU time remains billed. This tool does
    nothing unless confirm=true follows the user's explicit request.
    """
    if not confirm:
        return {
            "cancelled": False,
            "confirmation_required": True,
            "message": "Cancellation is irreversible; retry with confirm=true to proceed.",
        }
    return await runtime.sdk().models.delete_training(await runtime.owner(owner), project, model)


def register(mcp) -> None:
    for tool in (
        list_models,
        get_model,
        get_model_files,
        get_training_status,
        get_gpu_availability,
        start_training,
        cancel_training,
    ):
        mcp.tool(tool)
