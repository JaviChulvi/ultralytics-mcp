# Ultralytics Platform MCP

Work with your [Ultralytics Platform](https://platform.ultralytics.com) account from
your AI assistant in plain language — search the public catalog, import and edit
datasets, start and monitor training, export models, manage deployments and check
credits — without opening the web app. This is a hosted
[MCP](https://modelcontextprotocol.io) server: there is nothing to install.

## Quickstart (under 5 minutes)

1. **Create an API key** — go to [platform.ultralytics.com](https://platform.ultralytics.com)
   → **Settings → API Keys** and create a key (it starts with `ul_`).

2. **Register the server with your assistant** — for Claude Code, one command:

   ```bash
   claude mcp add --scope user --transport http ultralytics \
     https://mcp.ultralytics.com/mcp \
     --header "Authorization: Bearer ul_YOUR_KEY_HERE"
   ```

   Your key stays in your local configuration and is sent only with your own requests.

3. **Ask a question** — start your assistant and try:

   > what projects do I have on the platform?

If the key is missing or wrong, every tool answers with instructions on where to create
and configure one — nothing else is needed to troubleshoot.

To remove the server later:

```bash
claude mcp remove --scope user ultralytics
```

(`claude mcp list` shows what is registered and at which scope.)

## Tools

Every tool says in its description whether it is read-only or state-changing.
**Exactly one tool spends credits — `start_training` — and it refuses to run
without an explicit `confirm_spend=true`.** Deletes are soft (30-day trash) unless
marked permanent, and the permanent ones require `confirm=true`.

### Discover

| Tool | What it answers |
|---|---|
| `search_platform` | "Find me a public dataset for wildfire smoke" |
| `get_user_profile` | "Who is @ultralytics?" |

### Datasets

| Tool | What it does |
|---|---|
| `list_datasets` / `get_dataset` | Browse datasets; class stats and image statistics |
| `list_dataset_images` | Page through images (filter by split / labeled) |
| `list_dataset_models` | Models trained on a dataset (lineage) |
| `get_dataset_download` | Signed NDJSON download link (current or a version) |
| `create_dataset_version` | Immutable snapshot before risky edits |
| `create_dataset` / `update_dataset` / `delete_dataset` | Create, rename, trash |
| `import_dataset_from_url` | Import an archive URL into a new or existing dataset |

### Projects & models

| Tool | What it does |
|---|---|
| `list_projects` / `get_project` | Browse projects |
| `create_project` / `update_project` / `delete_project` | Create, rename, trash |
| `list_models` / `get_model` | Browse models and their metrics |
| `get_model_files` | Signed download links for trained weights |
| `update_model` / `delete_model` | Rename, trash |

### Training

| Tool | What it does |
|---|---|
| `get_gpu_availability` | GPU stock and hourly prices |
| `get_training_status` | Live epochs, progress and latest metrics |
| `start_training` | Start a cloud run — **spends credits, confirm-gated** |

### Exports & deployments

| Tool | What it does |
|---|---|
| `list_exports` / `get_export` | Browse exports; poll one until completed |
| `create_export` / `delete_export` | Export to ONNX/TensorRT/CoreML/... ; cancel or remove |
| `list_deployments` / `get_deployment` | Endpoints, health and latency |
| `create_deployment` | Deploy a model to a dedicated endpoint |
| `delete_deployment` | Remove an endpoint — **permanent, confirm-gated** |

### Account

| Tool | What it does |
|---|---|
| `get_account_status` | Credits, plan and storage |
| `get_recent_activity` | Recent account events |
| `list_trash` / `restore_from_trash` | What's expiring; bring items back |
| `purge_from_trash` | Free storage now — **permanent, confirm-gated** |

## Development

```bash
# one-time: install uv (https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync                    # reproducible environment from uv.lock
uv run pytest              # tests (platform API mocked)
uv run ruff check .        # lint
uv run uvicorn ultralytics_mcp.server:app --port 8000   # run locally
```

Optional checks:

```bash
ULTRALYTICS_TEST_API_KEY=ul_... uv run pytest -m live   # live smoke vs the real API
docker build -t ultralytics-mcp . && docker run -p 8000:8000 ultralytics-mcp
```

The live smoke suite is self-cleaning (it creates, trashes and purges its own
artifacts) and never starts a billed training run. Point it at a local platform
instance with `ULTRALYTICS_MCP_PLATFORM_BASE_URL=http://localhost:3002`.

For local testing, register the dev instance instead of the hosted URL:
`claude mcp add --transport http ultralytics-dev http://127.0.0.1:8000/mcp --header "Authorization: Bearer ul_YOUR_KEY"`.
Remove it with `claude mcp remove ultralytics-dev` when you're done.

### Project layout

```
src/ultralytics_mcp/
├── server.py            FastMCP app (stateless streamable HTTP at /mcp, /health)
├── auth.py              per-request Bearer token extraction — no credential storage
├── platform_client.py   one pooled httpx client; auth header set per request
├── errors.py            uniform, actionable error messages for every failure
├── schemas.py           whitelisted output models + response size cap (8 KB)
└── tools/               tools grouped by functionality: projects, datasets,
                         models, training, exports, deployments, account, discovery
tests/                   respx-mocked suite + opt-in live smoke tests
tests/fixtures/openapi.json   vendored upstream API contract snapshot
```

The upstream contract this code was written against is vendored at
`tests/fixtures/openapi.json`; re-download it from
`https://platform.ultralytics.com/openapi.json` and diff to spot upstream changes.
