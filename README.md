# Ultralytics Platform MCP

Ask your AI assistant about your [Ultralytics Platform](https://platform.ultralytics.com)
account in plain language — projects, datasets, models, training progress, deployments,
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

## Tools

All 13 tools are read-only and spend nothing.

| Tool | What it answers |
|---|---|
| `list_projects` | "What projects do I have?" |
| `get_project` | "Tell me about my warehouse-safety project" |
| `list_datasets` | "What datasets do I have?" |
| `get_dataset` | "What's in the forklifts dataset? What classes does it have?" |
| `list_dataset_images` | "Show me the training images of that dataset" |
| `list_models` | "Which models are in project X?" |
| `get_model` | "How good is my detector-v1 model?" |
| `get_training_status` | "How is my training run doing?" |
| `list_exports` | "Which formats did I export my model to?" |
| `list_deployments` | "What endpoints do I have running?" |
| `get_deployment` | "Is my production endpoint healthy? How's its latency?" |
| `get_account_status` | "How many credits do I have left? How much storage am I using?" |
| `get_recent_activity` | "What happened on my account this week?" |

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

For local testing, register the dev instance instead of the hosted URL:
`claude mcp add --transport http ultralytics-dev http://127.0.0.1:8000/mcp --header "Authorization: Bearer ul_YOUR_KEY"`.

### Project layout

```
src/ultralytics_mcp/
├── server.py            FastMCP app (stateless streamable HTTP at /mcp, /health)
├── auth.py              per-request Bearer token extraction — no credential storage
├── platform_client.py   one pooled httpx client; auth header set per request
├── errors.py            uniform, actionable error messages for every failure
├── schemas.py           whitelisted output models + response size cap (8 KB)
└── tools/               the 13 read-only tools, grouped by resource
tests/                   respx-mocked suite + opt-in live smoke tests
tests/fixtures/openapi.json   vendored upstream API contract snapshot
```

The upstream contract this code was written against is vendored at
`tests/fixtures/openapi.json`; re-download it from
`https://platform.ultralytics.com/openapi.json` and diff to spot upstream changes.
