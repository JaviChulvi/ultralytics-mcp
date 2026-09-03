from __future__ import annotations

import pytest
from fastmcp.exceptions import ToolError
from ultralytics_platform import APIConnectionError, APIError

from ultralytics_mcp.tools.models import cancel_training, start_training


async def test_training_requires_confirmation_without_sdk_mutation(sdk):
    result = await start_training("project", "data", "run", "Run", "rtx-4090")
    assert result["confirmation_required"] is True
    sdk.models.create.assert_not_awaited()
    sdk.training.start.assert_not_awaited()


async def test_training_creates_model_then_starts_compute(sdk):
    result = await start_training(
        "project",
        "data",
        "run",
        "Run",
        "rtx-4090",
        owner="team",
        dataset_owner="data-team",
        epochs=20,
        imgsz=640,
        extra_args={"patience": 10},
        confirm_spend=True,
    )
    train_args = {
        "patience": 10,
        "model": "yolo26n.pt",
        "data": "ul://data-team/datasets/data",
        "epochs": 20,
        "imgsz": 640,
    }
    sdk.models.create.assert_awaited_once_with(
        body={
            "owner": "team",
            "project": "project",
            "model": "run",
            "name": "Run",
            "trainArgs": train_args,
        }
    )
    sdk.training.start.assert_awaited_once_with(
        model_id="model-id",
        train_args=train_args,
        gpu_type="rtx-4090",
        capture_dataset_version=True,
    )
    assert result["model"]["id"] == "model-id"
    assert result["training"]["status"] == "starting"


async def test_training_rejects_reserved_extra_arguments(sdk):
    with pytest.raises(ToolError, match="reserved arguments: epochs"):
        await start_training(
            "project",
            "data",
            "run",
            "Run",
            "rtx-4090",
            extra_args={"epochs": 1},
            confirm_spend=True,
        )
    sdk.models.create.assert_not_awaited()


async def test_failed_start_cleans_up_created_model(sdk):
    sdk.training.start.side_effect = APIError(500, '{"error":"scheduler failed"}')
    with pytest.raises(ToolError, match="temporarily unavailable"):
        await start_training(
            "project",
            "data",
            "run",
            "Run",
            "rtx-4090",
            owner="team",
            confirm_spend=True,
        )
    sdk.models.delete.assert_awaited_once_with("team", "project", "run")


async def test_ambiguous_connection_failure_does_not_cancel_possible_training(sdk):
    sdk.training.start.side_effect = APIConnectionError("response lost")
    with pytest.raises(ToolError, match="outcome is unknown"):
        await start_training(
            "project",
            "data",
            "run",
            "Run",
            "rtx-4090",
            owner="team",
            confirm_spend=True,
        )
    sdk.models.delete.assert_not_awaited()


async def test_cancel_training_requires_confirmation(sdk):
    result = await cancel_training("project", "run", owner="team")
    assert result["confirmation_required"] is True
    sdk.models.delete_training.assert_not_awaited()
    await cancel_training("project", "run", owner="team", confirm=True)
    sdk.models.delete_training.assert_awaited_once_with("team", "project", "run")
