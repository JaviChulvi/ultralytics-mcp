# Ultralytics Platform MCP

A local [Model Context Protocol](https://modelcontextprotocol.io) server powered
exclusively by the official [`ultralytics-platform`](https://pypi.org/project/ultralytics-platform/)
Python SDK. It gives AI assistants a focused tool surface for discovering datasets,
training models, exporting artifacts, and operating managed deployments.

## Setup

Create an API key in [Ultralytics Platform settings](https://platform.ultralytics.com/settings?tab=api-keys),
then expose it to the process that launches the MCP:

```bash
export ULTRALYTICS_API_KEY=ul_your_key
uv sync
uv run ultralytics-mcp
```

The server uses stdio and stores no credentials. All Platform communication is made
by `AsyncPlatform`; this repository contains no raw API transport or copied OpenAPI
contract.

### Claude Code

```bash
claude mcp add --scope user --transport stdio \
  --env ULTRALYTICS_API_KEY="$ULTRALYTICS_API_KEY" \
  ultralytics -- uv --directory /absolute/path/to/ultralytics-mcp run ultralytics-mcp
```

Any MCP client that can launch a stdio command can use the same executable and pass
`ULTRALYTICS_API_KEY` in the child-process environment.

## Tools

The 26 tools cover the complete first-release workflow.

### Account and discovery

- `get_account_status` — get the authenticated workspace, plan, credits, storage,
  and resource counts.
- `search_platform` — search public Platform projects and datasets.

### Projects

- `list_projects` — list projects in a personal or team workspace.
- `get_project` — get a project and its model summaries.
- `create_project` — create a project in a personal or team workspace.

### Datasets

- `list_datasets` — list datasets, optionally including samples and image URLs.
- `get_dataset` — get dataset details and optional class and image statistics.
- `create_dataset` — create an empty dataset.
- `import_dataset_from_url` — ingest a remote archive or NDJSON file into a dataset.
- `get_dataset_download` — get a signed download URL for a dataset or saved version.
- `create_dataset_version` — create an immutable dataset snapshot.

### Models

- `list_models` — list models in a project.
- `get_model` — get model details or validation analysis.
- `get_model_files` — get signed download links for a model's files.

### Training

- `get_training_status` — get live training progress, metrics, and terminal status.
- `get_gpu_availability` — get current managed cloud GPU availability.
- `start_training` — create a model and start a confirmed, credit-spending cloud run.
- `cancel_training` — stop a confirmed active training run.

### Exports

- `list_exports` — list export jobs for a model.
- `create_export` — start an asynchronous model export.
- `get_export` — get export status and its download URL when complete.

### Deployments

- `list_deployments` — list managed inference endpoints.
- `create_deployment` — create a managed inference endpoint for a model.
- `get_deployment` — get deployment details, health, and metrics when available.
- `get_deployment_logs` — get recent deployment logs.
- `delete_deployment` — permanently delete a deployment after confirmation.

Training is the only credit-spending tool and will not start unless
`confirm_spend=true`. Cancellation and permanent deployment deletion likewise require
explicit confirmation.

Try prompts such as:

- “Show my datasets and inspect the class statistics for forklifts.”
- “Check my credits and GPU availability, then estimate a YOLO26n training run.”
- “Export warehouse/detector as ONNX and check whether the export is ready.”
- “List my deployments and investigate any unhealthy endpoint.”

## Development

```bash
uv sync
uv run ruff check .
uv run pytest
```

Live read-only smoke tests are opt-in:

```bash
ULTRALYTICS_TEST_API_KEY=ul_test_key uv run pytest -m live
```
