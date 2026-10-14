import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

SERVICE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SERVICE_ROOT not in sys.path:
    sys.path.insert(0, SERVICE_ROOT)

from src.llm_upstream import (  # noqa: E402
    mark_model_validated,
    reset_validation_cache_for_tests,
    upstream_post,
    validate_model_cached,
)


@pytest.fixture(autouse=True)
def _clear_caches():
    reset_validation_cache_for_tests()
    yield
    reset_validation_cache_for_tests()


def test_validation_cache_skips_validate_fn():
    mark_model_validated("qwen3-8-27", "chat", tool_id="agent_flow")
    validate_fn = AsyncMock(return_value=None)

    async def _run():
        return await validate_model_cached(
            validate_fn, "qwen3-8-27", "chat", tool_id="agent_flow"
        )

    err = asyncio.run(_run())
    assert err is None
    validate_fn.assert_not_called()


def test_validate_single_pass_on_cache_miss():
    validate_fn = AsyncMock(return_value=None)

    async def _run():
        return await validate_model_cached(validate_fn, "qwen3-8-27", "chat")

    err = asyncio.run(_run())
    assert err is None
    assert validate_fn.await_count == 1


def test_upstream_post_does_not_retry_http_503():
    busy = httpx.Response(503, json={"detail": "concurrency_timeout"})
    client = MagicMock()
    client.post = AsyncMock(return_value=busy)

    async def _run():
        return await upstream_post(client, "http://gw/v1/chat", content=b"{}", log_label="t")

    resp = asyncio.run(_run())
    assert resp.status_code == 503
    assert client.post.await_count == 1


def test_upstream_post_retries_connect_error_only():
    ok = httpx.Response(200, json={"ok": True})
    client = MagicMock()
    client.post = AsyncMock(
        side_effect=[httpx.ConnectError("refused"), ok],
    )

    async def _run():
        with patch("src.llm_upstream._infra_retry_delay_sec", return_value=0):
            return await upstream_post(client, "http://gw/v1/chat", content=b"{}", log_label="t")

    resp = asyncio.run(_run())
    assert resp.status_code == 200
    assert client.post.await_count == 2
