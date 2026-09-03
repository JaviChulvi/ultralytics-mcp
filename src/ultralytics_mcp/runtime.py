"""Lifespan-owned access to the official Ultralytics Platform SDK."""

from __future__ import annotations

import os

from fastmcp.exceptions import ToolError
from ultralytics_platform import AsyncPlatform

CREDENTIAL_GUIDANCE = (
    "ULTRALYTICS_API_KEY is not set. Create an API key at "
    "https://platform.ultralytics.com/settings?tab=api-keys, export it in the "
    "environment that launches this MCP server, and try again."
)


class PlatformRuntime:
    """Own the SDK client and resolve the current personal workspace lazily."""

    def __init__(self) -> None:
        self.client: AsyncPlatform | None = None
        self._default_owner: str | None = None

    async def start(self) -> None:
        if not os.environ.get("ULTRALYTICS_API_KEY"):
            raise RuntimeError(CREDENTIAL_GUIDANCE)
        self.client = AsyncPlatform()

    async def close(self) -> None:
        if self.client is not None:
            await self.client.close()
        self.client = None
        self._default_owner = None

    def sdk(self) -> AsyncPlatform:
        if self.client is None:
            raise ToolError("The Ultralytics Platform SDK is not initialized.")
        return self.client

    async def owner(self, explicit: str | None = None) -> str:
        if explicit:
            return explicit
        if self._default_owner is None:
            summary = await self.sdk().account.summary()
            self._default_owner = summary["username"]
        return self._default_owner


runtime = PlatformRuntime()
