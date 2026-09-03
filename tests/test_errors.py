from __future__ import annotations

from ultralytics_platform import APIConnectionError, APIError

from ultralytics_mcp.errors import translate


def test_structured_sdk_error_is_actionable_and_keeps_request_id():
    error = APIError(409, '{"error":"Export already running"}', "request-123")
    message = str(translate(error))
    assert "Export already running" in message
    assert "request-123" in message
    assert "ul_secret" not in message


def test_invalid_credentials_do_not_echo_sdk_body():
    error = APIError(401, '{"error":"bad ul_secret_key"}')
    message = str(translate(error))
    assert "ULTRALYTICS_API_KEY" in message
    assert "ul_secret_key" not in message


def test_connection_failure_is_safe():
    message = str(translate(APIConnectionError("socket contained ul_secret")))
    assert "outcome is unknown" in message
    assert "ul_secret" not in message


def test_upstream_reason_redacts_key_like_values():
    error = APIError(400, '{"error":"bad token ul_actual_secret"}')
    message = str(translate(error))
    assert "[REDACTED]" in message
    assert "ul_actual_secret" not in message
