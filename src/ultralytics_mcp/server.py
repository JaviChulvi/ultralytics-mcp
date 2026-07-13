"""FastMCP server: hosted, stateless, streamable-HTTP MCP endpoint at /mcp (FR-001).

Run with: uvicorn ultralytics_mcp.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from . import __version__
from .settings import settings

INSTRUCTIONS = (
    "Tools for the Ultralytics Platform (platform.ultralytics.com): search the public "
    "catalog; browse, import and edit datasets; manage projects, models, exports and "
    "deployments; monitor training and endpoint health; and check account status. "
    "Every tool's description says whether it is read-only or state-changing. No tool "
    "spends credits unless its description says so explicitly, and anything that "
    "would spend requires an explicit confirmation parameter. Every request must "
    "carry the user's own platform API key as an 'Authorization: Bearer ul_...' "
    "HTTP header."
)

mcp = FastMCP(
    name="Ultralytics Platform",
    instructions=INSTRUCTIONS,
    version=__version__,
    mask_error_details=True,
)


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "version": __version__})


_tools_registered = False


def _register_tools() -> None:
    global _tools_registered
    if _tools_registered:
        return
    _tools_registered = True
    from .tools import (
        account,
        datasets,
        deployments,
        discovery,
        exports,
        models,
        projects,
        training,
    )

    projects.register(mcp)
    datasets.register(mcp)
    models.register(mcp)
    training.register(mcp)
    exports.register(mcp)
    deployments.register(mcp)
    account.register(mcp)
    discovery.register(mcp)


def create_app():
    logging.basicConfig(level=settings.log_level)
    _register_tools()
    return mcp.http_app(path="/mcp", stateless_http=True)


app = create_app()
