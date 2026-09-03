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

The 26 tools cover the complete first-release workflow:

- Account and discovery: `get_account_status`, `search_platform`
- Projects: `list_projects`, `get_project`, `create_project`
- Datasets: `list_datasets`, `get_dataset`, `create_dataset`,
  `import_dataset_from_url`, `get_dataset_download`, `create_dataset_version`
- Models and training: `list_models`, `get_model`, `get_model_files`,
  `get_training_status`, `get_gpu_availability`, `start_training`, `cancel_training`
- Exports: `list_exports`, `create_export`, `get_export`
- Deployments: `list_deployments`, `create_deployment`, `get_deployment`,
  `get_deployment_logs`, `delete_deployment`

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
