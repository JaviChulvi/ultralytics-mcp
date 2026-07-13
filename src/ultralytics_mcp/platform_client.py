"""Shared async HTTP client for the Ultralytics Platform API.

One pooled ``httpx.AsyncClient`` for the whole process; the Authorization header is
set per request and never on the client, so concurrent users can never share or leak
a credential (FR-002). The server stores nothing (FR-011).
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from .errors import PlatformError
from .settings import settings


class PlatformClient:
    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        # Pooled connections are bound to the loop that created them; recreate the
        # client if the running loop changed (test runners create one per test).
        loop = asyncio.get_running_loop()
        if self._client is None or self._client.is_closed or self._loop is not loop:
            self._client = httpx.AsyncClient(
                base_url=settings.platform_base_url,
                timeout=httpx.Timeout(settings.read_timeout, connect=settings.connect_timeout),
            )
            self._loop = loop
        return self._client

    async def request(
        self,
        method: str,
        path: str,
        *,
        token: str,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        resource_hint: str | None = None,
    ) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {token}"}
        try:
            response = await self.client.request(
                method, path, params=params, json=json, headers=headers
            )
        except httpx.HTTPError as exc:
            raise PlatformError(None, resource_hint) from exc
        if response.status_code >= 400:
            # Error bodies carry actionable structure (error text, quota numbers,
            # in-flight job ids) that the translation layer surfaces to the user.
            try:
                detail = response.json()
            except ValueError:
                detail = None
            raise PlatformError(response.status_code, resource_hint, detail=detail)
        return response.json()

    async def get(
        self,
        path: str,
        *,
        token: str,
        params: dict[str, Any] | None = None,
        resource_hint: str | None = None,
    ) -> dict[str, Any]:
        return await self.request(
            "GET", path, token=token, params=params, resource_hint=resource_hint
        )

    async def post(
        self,
        path: str,
        *,
        token: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        resource_hint: str | None = None,
    ) -> dict[str, Any]:
        return await self.request(
            "POST", path, token=token, params=params, json=json, resource_hint=resource_hint
        )

    async def patch(
        self,
        path: str,
        *,
        token: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        resource_hint: str | None = None,
    ) -> dict[str, Any]:
        return await self.request(
            "PATCH", path, token=token, params=params, json=json, resource_hint=resource_hint
        )

    async def delete(
        self,
        path: str,
        *,
        token: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        resource_hint: str | None = None,
    ) -> dict[str, Any]:
        return await self.request(
            "DELETE", path, token=token, params=params, json=json, resource_hint=resource_hint
        )


platform_api = PlatformClient()
