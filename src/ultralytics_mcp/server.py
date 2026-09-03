"""Stdio entry point for the SDK-first Ultralytics Platform MCP."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastmcp import FastMCP

from . import __version__
from .runtime import PlatformRuntime, runtime
from .tools import register_tools

INSTRUCTIONS = """\
Use these tools to work with the Ultralytics Platform through the official Python SDK.
ULTRALYTICS_API_KEY identifies the personal workspace; pass owner explicitly for a
team or public workspace. Read tool descriptions before mutations. Cloud training
spends credits and requires confirm_spend=true. Training cancellation and permanent
deployment deletion require confirm=true. Poll status tools for asynchronous work.
"""


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[PlatformRuntime]:
    del server
    await runtime.start()
    try:
        yield runtime
    finally:
        await runtime.close()


mcp = FastMCP(
    name="Ultralytics Platform",
    instructions=INSTRUCTIONS,
    version=__version__,
    lifespan=lifespan,
    mask_error_details=True,
)
register_tools(mcp)


def main() -> None:
    """Run the local MCP over stdio."""
    mcp.run(transport="stdio", show_banner=False)


if __name__ == "__main__":
    main()
