from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ultralytics_mcp.runtime import runtime


@pytest.fixture
def sdk():
    client = SimpleNamespace(
        account=SimpleNamespace(
            summary=AsyncMock(return_value={"username": "test-user"}),
            storage=AsyncMock(return_value={"usage": {}}),
        ),
        explore=SimpleNamespace(search=AsyncMock(return_value={"results": []})),
        projects=SimpleNamespace(
            list=AsyncMock(return_value={"projects": []}),
            retrieve=AsyncMock(return_value={"project": {}}),
            create=AsyncMock(return_value={"project": "demo"}),
        ),
        datasets=SimpleNamespace(
            list=AsyncMock(return_value={"datasets": []}),
            retrieve=AsyncMock(return_value={"dataset": {}}),
            class_stats=AsyncMock(return_value={"classes": []}),
            create=AsyncMock(return_value={"dataset": "demo"}),
            ingest=AsyncMock(return_value={"jobId": "job-1", "status": "queued"}),
            export=AsyncMock(return_value={"downloadUrl": "https://example.test/data"}),
            create_export=AsyncMock(return_value={"version": 1}),
        ),
        models=SimpleNamespace(
            list=AsyncMock(return_value={"models": []}),
            retrieve=AsyncMock(return_value={"model": {}}),
            files=AsyncMock(return_value={"files": []}),
            training=AsyncMock(return_value={"status": "running"}),
            create=AsyncMock(
                return_value={
                    "id": "model-id",
                    "owner": "test-user",
                    "project": "demo",
                    "model": "run-1",
                    "region": "eu",
                }
            ),
            delete=AsyncMock(return_value={"success": True}),
            delete_training=AsyncMock(return_value={"status": "cancelled"}),
        ),
        training=SimpleNamespace(
            gpu_availability=AsyncMock(return_value={"rtx-4090": "High"}),
            start=AsyncMock(
                return_value={
                    "modelId": "model-id",
                    "status": "starting",
                    "gpuType": "rtx-4090",
                }
            ),
        ),
        exports=SimpleNamespace(
            list=AsyncMock(return_value={"exports": []}),
            create=AsyncMock(return_value={"exportId": "export-1"}),
            retrieve=AsyncMock(return_value={"export": {"status": "completed"}}),
        ),
        deployments=SimpleNamespace(
            list=AsyncMock(return_value={"deployments": []}),
            create=AsyncMock(return_value={"deployment": "prod"}),
            retrieve=AsyncMock(return_value={"deployment": {"status": "ready"}}),
            health=AsyncMock(return_value={"healthy": True}),
            metrics=AsyncMock(return_value={"summary": {}}),
            logs=AsyncMock(return_value={"entries": []}),
            delete=AsyncMock(return_value={"success": True}),
        ),
    )
    old_client, old_owner = runtime.client, runtime._default_owner
    runtime.client, runtime._default_owner = client, "test-user"
    try:
        yield client
    finally:
        runtime.client, runtime._default_owner = old_client, old_owner
