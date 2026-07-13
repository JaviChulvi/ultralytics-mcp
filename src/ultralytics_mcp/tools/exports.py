"""Export tools (US3, FR-004): list, create, poll and remove model format exports.

Exports never spend credits — the only gates are plan tier (some TensorRT GPUs)
and a 20/min rate limit.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastmcp.exceptions import ToolError
from pydantic import Field

from ..auth import get_request_token
from ..errors import platform_errors
from ..platform_client import platform_api
from ..schemas import ExportSummary, clamp_limit, make_list_result
from .models import _resolve_model

EXPORT_FORMATS = (
    "onnx",
    "torchscript",
    "openvino",
    "engine",
    "coreml",
    "litert",
    "pb",
    "saved_model",
    "paddle",
    "ncnn",
    "edgetpu",
    "mnn",
    "rknn",
    "qnn",
    "imx",
    "axelera",
    "executorch",
    "deepx",
)


@platform_errors
async def list_exports(
    model_id: Annotated[
        str, Field(description="The source model's id (24-char hex); the platform requires it")
    ],
    status: Annotated[str | None, Field(description="Filter by export status")] = None,
    limit: Annotated[
        int | None, Field(description="Max exports to return (default 20, max 50)", ge=1)
    ] = None,
) -> dict[str, Any]:
    """List a model's exports (format conversions like ONNX, TensorRT, CoreML).

    Read-only — spends nothing. Requires the model's id (find it with list_models).
    Returns each export's id, format, status, artifact file name and completion time.
    """
    token = get_request_token()
    params: dict[str, Any] = {"limit": clamp_limit(limit), "modelId": model_id}
    if status:
        params["status"] = status
    data = await platform_api.get(
        "/api/exports", token=token, params=params, resource_hint="Your export list"
    )
    exports = [ExportSummary.from_api(item) for item in data.get("exports", [])]
    return make_list_result(exports, empty_note="No exports yet — create one with create_export.")


@platform_errors
async def create_export(
    model: Annotated[str, Field(description="Model id (24-char hex) or slug — must be trained")],
    format: Annotated[
        str,
        Field(description=f"Target format, one of: {', '.join(EXPORT_FORMATS)}"),
    ],
    gpu_type: Annotated[
        str | None,
        Field(
            description="GPU for TensorRT ('engine') exports only, e.g. 'rtx-4090' — "
            "the engine is built on and optimized for this GPU"
        ),
    ] = None,
    imgsz: Annotated[
        int | None, Field(description="Export image size (default: the model's own)", ge=32)
    ] = None,
    quantize: Annotated[
        str | None,
        Field(description="Precision: 8/int8, 16/fp16, 32/fp32, w8a16 or w8a32"),
    ] = None,
    dynamic: Annotated[bool | None, Field(description="Dynamic input shapes")] = None,
    simplify: Annotated[bool | None, Field(description="Simplify the ONNX graph")] = None,
    opset: Annotated[int | None, Field(description="ONNX opset (9-23)", ge=9, le=23)] = None,
    batch: Annotated[int | None, Field(description="Batch size (1-32)", ge=1, le=32)] = None,
    nms: Annotated[bool | None, Field(description="Embed NMS in the exported model")] = None,
    extra_args: Annotated[
        dict[str, Any] | None,
        Field(description="Other export args the platform accepts (validated upstream)"),
    ] = None,
) -> dict[str, Any]:
    """Export a trained model to a deployment format (ONNX, TensorRT, CoreML, ...).

    State-changing — spends no credits (some TensorRT GPUs need a Pro plan). The
    export runs asynchronously: poll get_export until status is 'completed', then
    download via its signed URL. A conflict means the same format is already being
    exported — the error names the in-flight export id.
    """
    if format not in EXPORT_FORMATS:
        raise ToolError(f"Unknown format '{format}' — choose from: {', '.join(EXPORT_FORMATS)}.")
    if format == "engine" and not gpu_type:
        raise ToolError(
            "TensorRT ('engine') exports build on a specific GPU — pass gpu_type "
            "(e.g. 'rtx-4090'; see get_gpu_availability)."
        )
    token = get_request_token()
    resolved = await _resolve_model(token, model)
    if isinstance(resolved, dict):
        return resolved
    args = {
        key: value
        for key, value in {
            "imgsz": imgsz,
            "quantize": quantize,
            "dynamic": dynamic,
            "simplify": simplify,
            "opset": opset,
            "batch": batch,
            "nms": nms,
            **(extra_args or {}),
        }.items()
        if value is not None
    }
    body: dict[str, Any] = {"modelId": resolved, "format": format}
    if gpu_type:
        body["gpuType"] = gpu_type
    if args:
        body["args"] = args
    data = await platform_api.post(
        "/api/exports", token=token, json=body, resource_hint=f"Export of model '{model}'"
    )
    return {
        "export_id": data.get("exportId"),
        "format": data.get("format", format),
        "status": data.get("status"),
        "note": "Export queued — poll get_export until status is 'completed' "
        "(TensorRT builds can take several minutes).",
    }


@platform_errors
async def get_export(
    export_id: Annotated[str, Field(description="Export id from create_export/list_exports")],
) -> dict[str, Any]:
    """Check one export's progress and get its download link when done.

    Read-only apart from platform-side completion bookkeeping — spends nothing.
    This is the polling endpoint: for TensorRT jobs the platform finalizes
    completion during this call. A completed export includes a signed download URL.
    """
    token = get_request_token()
    data = await platform_api.get(
        f"/api/exports/{export_id}", token=token, resource_hint=f"Export '{export_id}'"
    )
    summary = ExportSummary.from_api(data.get("export") or {})
    result = summary.model_dump(exclude_none=True)
    if summary.status in ("queued", "starting", "running"):
        result["note"] = "Still in progress — poll again shortly."
    elif summary.status == "completed" and summary.download_url:
        result["note"] = "Ready — the download URL is signed and short-lived."
    return result


@platform_errors
async def delete_export(
    export_id: Annotated[str, Field(description="Export id from create_export/list_exports")],
) -> dict[str, Any]:
    """Cancel a running export, or delete a finished one and its artifact.

    State-changing — spends no credits. An in-progress job is cancelled; a finished
    export is removed along with its stored file (freeing storage). Re-exporting
    later recreates it.
    """
    token = get_request_token()
    data = await platform_api.delete(
        f"/api/exports/{export_id}", token=token, resource_hint=f"Export '{export_id}'"
    )
    action = data.get("action")
    note = (
        "In-progress export cancelled."
        if action == "cancelled"
        else "Export and its artifact deleted — re-export to recreate it."
    )
    return {"success": bool(data.get("success", True)), "action": action, "note": note}


def register(mcp) -> None:
    mcp.tool(list_exports)
    mcp.tool(create_export)
    mcp.tool(get_export)
    mcp.tool(delete_export)
