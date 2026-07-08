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

All tools are read-only and spend nothing.

| Tool | What it answers |
|---|---|
| `list_projects` | "What projects do I have?" |

More visibility tools (datasets, models, training progress, deployments, account) land
with the next user stories of this feature.

## Development

```bash
# one-time: install uv (https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync                    # reproducible environment from uv.lock
uv run pytest              # tests (platform API mocked)
uv run ruff check .        # lint
uv run uvicorn ultralytics_mcp.server:app --port 8000   # run locally
```

For local testing, register the dev instance instead of the hosted URL:
`claude mcp add --transport http ultralytics-dev http://127.0.0.1:8000/mcp --header "Authorization: Bearer ul_YOUR_KEY"`.
