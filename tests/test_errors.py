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


DETAIL_CASES = [
    pytest.param(
        402,
        {"error": "Insufficient balance for this training run."},
        ["insufficient balance", "top up", "billing"],
        id="402-insufficient-balance",
    ),
    pytest.param(
        403,
        {"error": "Quota exceeded", "quotaType": "deployments", "current": 3, "limit": 3},
        ["deployments quota", "3/3", "upgrade"],
        id="403-quota",
    ),
    pytest.param(
        403,
        {"error": "B200 GPUs require a Pro plan.", "code": "GPU_TIER_RESTRICTED"},
        ["b200", "pro plan", "upgrade"],
        id="403-tier-gate",
    ),
    pytest.param(
        409,
        {"error": "Dataset is already processing.", "existingJobId": "job_42"},
        ["already processing", "job_42", "retry"],
        id="409-in-flight-job",
    ),
    pytest.param(
        400,
        {"error": "trainArgs.epochs must be between 1 and 10000"},
        ["http 400", "epochs must be between"],
        id="400-upstream-reason",
    ),
]


@pytest.mark.parametrize(("status", "detail", "expected"), DETAIL_CASES)
def test_translate_surfaces_structured_detail(status, detail, expected):
    message = str(translate(PlatformError(status, "Dataset 'x'", detail=detail))).lower()
    for fragment in expected:
        assert fragment in message, f"expected {fragment!r} in {message!r}"


def test_translate_ignores_non_dict_detail():
    message = str(translate(PlatformError(400, None, detail=["not", "a", "dict"])))
    assert "HTTP 400" in message
