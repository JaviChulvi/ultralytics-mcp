"""Shared async HTTP client for the Ultralytics Platform API.

One pooled ``httpx.AsyncClient`` for the whole process; the Authorization header is
set per request and never on the client, so concurrent users can never share or leak
a credential (FR-002). The server stores nothing (FR-011).
"""

from __future__ import annotations

from typing import Any

import httpx

from .errors import PlatformError
from .settings import settings


class PlatformClient:
    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=settings.platform_base_url,
                timeout=httpx.Timeout(settings.read_timeout, connect=settings.connect_timeout),
            )
        return self._client

    async def request(
        self,
        method: str,
        path: str,
        *,
        token: str,
        params: dict[str, Any] | None = None,
        resource_hint: str | None = None,
    ) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {token}"}
        try:
            response = await self.client.request(method, path, params=params, headers=headers)
        except httpx.HTTPError as exc:
            raise PlatformError(None, resource_hint) from exc
        if response.status_code >= 400:
            raise PlatformError(response.status_code, resource_hint)
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


platform_api = PlatformClient()
