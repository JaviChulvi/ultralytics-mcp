"""Register the curated Ultralytics Platform tool surface."""

from __future__ import annotations


def register_tools(mcp) -> None:
    from . import account, datasets, deployments, exports, models, projects

    for module in (account, projects, datasets, models, exports, deployments):
        module.register(mcp)
