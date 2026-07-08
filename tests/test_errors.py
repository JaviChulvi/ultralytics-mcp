"""Every mapped failure names its cause and a next step; nothing internal leaks (SC-004)."""

from __future__ import annotations

import pytest
from fastmcp.exceptions import ToolError

from ultralytics_mcp.errors import CREDENTIAL_GUIDANCE, PlatformError, translate

TEST_TOKEN = "ul_test_0123456789abcdef"  # matches conftest; must never appear in messages

CASES = [
    pytest.param(401, None, ["api key", "settings > api keys"], id="401-credential"),
    pytest.param(403, "Project 'x'", ["access", "account"], id="403-forbidden"),
    pytest.param(
        404, "Dataset 'foo'", ["dataset 'foo'", "not found", "list tools"], id="404-not-found"
    ),
    pytest.param(429, None, ["rate-limiting", "retrying is safe"], id="429-throttled"),
    pytest.param(500, None, ["unavailable", "retrying later is safe"], id="500-server-error"),
    pytest.param(503, None, ["unavailable"], id="503-unavailable"),
    pytest.param(None, None, ["could not reach", "retrying later is safe"], id="network-timeout"),
    pytest.param(418, None, ["rejected", "http 418"], id="unmapped-status"),
]


@pytest.mark.parametrize(("status", "hint", "expected"), CASES)
def test_translate_names_cause_and_next_step(status, hint, expected):
    error = translate(PlatformError(status, hint))
    assert isinstance(error, ToolError)
    message = str(error).lower()
    for fragment in expected:
        assert fragment in message, f"expected {fragment!r} in {message!r}"


@pytest.mark.parametrize(("status", "hint", "expected"), CASES)
def test_translate_leaks_nothing_internal(status, hint, expected):
    message = str(translate(PlatformError(status, hint)))
    assert "Traceback" not in message
    assert TEST_TOKEN not in message
    assert "PlatformError" not in message


def test_credential_guidance_names_key_location():
    assert "Settings > API Keys" in CREDENTIAL_GUIDANCE
    assert "platform.ultralytics.com" in CREDENTIAL_GUIDANCE
