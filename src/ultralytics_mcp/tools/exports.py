"""Model export tools backed directly by the official SDK."""

from __future__ import annotations

from typing import Any, Literal

from ..errors import sdk_errors
from ..runtime import runtime
from .common import provided

ExportFormat = Literal[
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
    "hailo",
    "ascend",
]


@sdk_errors
async def list_exports(
    project: str,
    model: str,
    owner: str | None = None,
    status: Literal["queued", "starting", "running", "completed", "failed", "cancelled"]
    | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """List export jobs for a model. Read-only and spends nothing."""
    return await runtime.sdk().exports.list(
        await runtime.owner(owner),
        project,
        model,
        **provided(status=status, limit=max(1, min(limit, 50))),
    )


@sdk_errors
async def create_export(
    project: str,
    model: str,
    format: ExportFormat,
    owner: str | None = None,
    gpu_type: str | None = None,
    args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Start an asynchronous model export. State-changing but spends no credits.

    TensorRT `engine` exports should specify the exact target GPU type. Poll
    get_export until the job completes.
    """
    return await runtime.sdk().exports.create(
        await runtime.owner(owner),
        project,
        model,
        **provided(format=format, gpu_type=gpu_type, args=args),
    )


@sdk_errors
async def get_export(
    project: str, model: str, export_id: str, owner: str | None = None
) -> dict[str, Any]:
    """Get an export's status and completed download URL. Read-only and spends nothing."""
    return await runtime.sdk().exports.retrieve(
        await runtime.owner(owner), project, model, export_id
    )


def register(mcp) -> None:
    mcp.tool(list_exports)
    mcp.tool(create_export)
    mcp.tool(get_export)
