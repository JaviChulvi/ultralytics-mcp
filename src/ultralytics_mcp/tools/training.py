"""Training tools (US3): GPU availability/pricing and starting cloud training runs.

start_training is the ONLY tool in this server that spends credits. It refuses to
run without confirm_spend=true, and its description spells out how billing works.
Cancelling a run currently requires the web UI — the platform's cancel endpoint
does not accept API keys yet (raised with the platform team).
"""

from __future__ import annotations

from typing import Annotated, Any

from fastmcp.exceptions import ToolError
from pydantic import Field

from ..auth import get_request_token
from ..errors import PlatformError, platform_errors
from ..platform_client import platform_api
from .datasets import _resolve_dataset

# Platform GPU catalog (packages/ui platform-gpus.ts), snapshot 2026-07. Prices are
# USD/hour as listed by the platform; availability is fetched live. The platform UI
# is authoritative if these drift.
GPU_PRICING: dict[str, dict[str, Any]] = {
    "rtx-2000-ada": {"name": "RTX 2000 Ada", "price_per_hour_usd": 0.24},
    "rtx-a4500": {"name": "RTX A4500", "price_per_hour_usd": 0.25},
    "rtx-4000-ada": {"name": "RTX 4000 Ada", "price_per_hour_usd": 0.26},
    "rtx-a5000": {"name": "RTX A5000", "price_per_hour_usd": 0.27},
    "l4": {"name": "L4", "price_per_hour_usd": 0.39},
    "a40": {"name": "A40", "price_per_hour_usd": 0.44},
    "rtx-3090": {"name": "RTX 3090", "price_per_hour_usd": 0.46},
    "rtx-a6000": {"name": "RTX A6000", "price_per_hour_usd": 0.49},
    "rtx-pro-4000": {"name": "RTX PRO 4000", "price_per_hour_usd": 0.57},
    "rtx-pro-4500": {"name": "RTX PRO 4500", "price_per_hour_usd": 0.64},
    "rtx-4090": {"name": "RTX 4090", "price_per_hour_usd": 0.69},
    "rtx-6000-ada": {"name": "RTX 6000 Ada", "price_per_hour_usd": 0.77},
    "l40s": {"name": "L40S", "price_per_hour_usd": 0.86},
    "rtx-pro-5000": {"name": "RTX PRO 5000", "price_per_hour_usd": 0.96},
    "rtx-5090": {"name": "RTX 5090", "price_per_hour_usd": 0.99},
    "l40": {"name": "L40", "price_per_hour_usd": 0.99},
    "a100-80gb-pcie": {"name": "A100 PCIe", "price_per_hour_usd": 1.39},
    "a100-80gb-sxm": {"name": "A100 SXM", "price_per_hour_usd": 1.49},
    "rtx-pro-6000": {"name": "RTX PRO 6000", "price_per_hour_usd": 2.09},
    "h100-pcie": {"name": "H100 PCIe", "price_per_hour_usd": 2.89},
    "h100-nvl": {"name": "H100 NVL", "price_per_hour_usd": 3.19},
    "h100-sxm": {"name": "H100 SXM", "price_per_hour_usd": 3.29},
    "h200-nvl": {"name": "H200 NVL", "price_per_hour_usd": 3.39},
    "h200-sxm": {"name": "H200 SXM", "price_per_hour_usd": 4.39},
    "b200": {"name": "B200", "price_per_hour_usd": 5.89, "min_tier": "pro"},
    "b300": {"name": "B300", "price_per_hour_usd": 7.39, "min_tier": "pro"},
}


@platform_errors
async def get_gpu_availability() -> dict[str, Any]:
    """See which training GPUs are in stock, with their hourly prices.

    Read-only — spends nothing. Returns each cloud GPU's listed USD/hour rate and
    live stock level (High/Medium/Low, null = unknown). B200/B300 require a Pro
    plan. Prices are the platform's listed rates at packaging time — the platform
    UI is authoritative if they drift.
    """
    token = get_request_token()
    stock = await platform_api.get(
        "/api/training/gpu-availability", token=token, resource_hint="GPU availability"
    )
    gpus = [
        {"id": gpu_id, **info, "availability": stock.get(gpu_id)}
        for gpu_id, info in GPU_PRICING.items()
    ]
    return {
        "gpus": gpus,
        "note": "Pass a GPU's id as start_training's gpu_type. Cost scales with "
        "epochs, image count and image size; the start_training response includes "
        "the platform's own estimate.",
    }


def _require_spend_confirmation(gpu_type: str) -> None:
    price = GPU_PRICING.get(gpu_type, {}).get("price_per_hour_usd")
    rate = f" at ~${price}/h" if price else ""
    raise ToolError(
        "start_training SPENDS CREDITS: the run bills the account per GPU-minute"
        f"{rate} ({gpu_type}), metered while it runs and settled on completion. "
        "Confirm with the user, then call again with confirm_spend=true. Note: "
        "cancelling mid-run currently requires the web UI (the model page's Stop "
        "button) — the cancel API does not accept API keys yet."
    )


@platform_errors
async def start_training(
    dataset: Annotated[
        str,
        Field(description="Platform dataset to train on: id, slug or 'username/slug'"),
    ],
    project: Annotated[str, Field(description="Project (id or slug) the new model is created in")],
    base_model: Annotated[
        str,
        Field(
            description="Official weights like 'yolo26n.pt'/'yolo11s.pt', or a "
            "'ul://user/project/model' URI to fine-tune your own model"
        ),
    ] = "yolo26n.pt",
    epochs: Annotated[int, Field(description="Training epochs (1-10000)", ge=1, le=10000)] = 100,
    gpu_type: Annotated[
        str, Field(description="GPU id from get_gpu_availability (default rtx-4090)")
    ] = "rtx-4090",
    imgsz: Annotated[
        int | None, Field(description="Training image size (default 640)", ge=32, le=4096)
    ] = None,
    batch: Annotated[int | None, Field(description="Batch size (-1 = auto)", ge=-1, le=512)] = None,
    model_name: Annotated[
        str | None, Field(description="Name for the new model (default: exp, exp-2, ...)")
    ] = None,
    extra_args: Annotated[
        dict[str, Any] | None,
        Field(description="Other YOLO train args (optimizer, patience, lr0, ...)"),
    ] = None,
    confirm_spend: Annotated[
        bool,
        Field(
            description="Must be true to start — training BILLS CREDITS per "
            "GPU-minute; confirm the cost with the user first"
        ),
    ] = False,
) -> dict[str, Any]:
    """Start a cloud GPU training run — THE ONLY TOOL HERE THAT SPENDS CREDITS.

    Bills the account per GPU-minute (metered live, settled at the end, ~15-minute
    minimum). Refuses to run without confirm_spend=true. Creates a model in the
    project and starts training it on the dataset; the response echoes the
    platform's cost estimate and your balance. Poll get_training_status for
    progress. Cancelling mid-run currently requires the web UI (model page > Stop);
    delete_model also cancels, billing the elapsed time.
    """
    if not confirm_spend:
        _require_spend_confirmation(gpu_type)
    if not (base_model.endswith(".pt") or base_model.startswith("ul://")):
        raise ToolError(
            "base_model must be official .pt weights (e.g. 'yolo26n.pt') or a "
            "'ul://user/project/model' URI."
        )
    token = get_request_token()

    resolved, owner, detail = await _resolve_dataset(token, dataset)
    if isinstance(resolved, list):
        from .datasets import _candidates

        return _candidates(resolved, dataset)
    if detail is None:
        detail = await platform_api.get(
            f"/api/datasets/{resolved}",
            token=token,
            params={"username": owner} if owner else None,
            resource_hint=f"Dataset '{dataset}'",
        )
    raw = detail.get("dataset") or {}
    ds_username, ds_slug = raw.get("username"), raw.get("slug")
    if not ds_username or not ds_slug:
        raise ToolError(f"Dataset '{dataset}' has no owner/slug — cannot build its data URI.")
    data_uri = f"ul://{ds_username}/datasets/{ds_slug}"

    from .projects import _resolve_project

    project_id = await _resolve_project(token, project)
    if isinstance(project_id, dict):
        return project_id

    shell_body: dict[str, Any] = {"projectId": project_id}
    if model_name:
        shell_body["name"] = model_name
    if raw.get("task"):
        shell_body["task"] = raw["task"]
    shell = await platform_api.post(
        "/api/models", token=token, json=shell_body, resource_hint="New model"
    )
    model_id = str(shell.get("modelId", ""))

    train_args: dict[str, Any] = {"model": base_model, "data": data_uri, "epochs": epochs}
    if imgsz is not None:
        train_args["imgsz"] = imgsz
    if batch is not None:
        train_args["batch"] = batch
    if extra_args:
        train_args.update(extra_args)
    try:
        started = await platform_api.post(
            "/api/training/start",
            token=token,
            json={"modelId": model_id, "gpuType": gpu_type, "trainArgs": train_args},
            resource_hint="Training start",
        )
    except PlatformError:
        # Don't leave an orphaned 'pending' shell behind when nothing started.
        try:
            await platform_api.delete(f"/api/models/{model_id}", token=token)
        except PlatformError:
            pass
        raise

    billing = started.get("billing") or {}
    estimate = started.get("estimatedCost") or {}
    result: dict[str, Any] = {
        "model_id": started.get("modelId", model_id),
        "model_slug": shell.get("slug"),
        "status": started.get("status", "starting"),
        "gpu_type": started.get("gpuType", gpu_type),
        "price_per_hour_usd": estimate.get("pricePerHour"),
        "estimated_cost": billing.get("estimatedCostDisplay"),
        "balance_cents": billing.get("balanceCents"),
        "note": "Training started and billing per GPU-minute. Poll "
        "get_training_status for progress. To stop early, use the model page's "
        "Stop button in the web UI (or delete_model, which cancels and bills the "
        "elapsed time).",
    }
    if billing.get("estimatedCostCents") is not None:
        result["estimated_cost_cents"] = billing["estimatedCostCents"]
    return result


def register(mcp) -> None:
    mcp.tool(get_gpu_availability)
    mcp.tool(start_training)
