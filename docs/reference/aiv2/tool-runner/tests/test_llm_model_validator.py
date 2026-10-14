import asyncio
import os
import sys
import time
from unittest.mock import AsyncMock, patch

import pytest

SERVICE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SERVICE_ROOT not in sys.path:
    sys.path.insert(0, SERVICE_ROOT)

from src.llm_model_validator import (  # noqa: E402
    _catalog_cache,
    _fetch_catalog,
    reset_catalog_cache_for_tests,
    validate_model_for_profile,
)


@pytest.fixture(autouse=True)
def _clear_catalog_cache():
    reset_catalog_cache_for_tests()
    yield
    reset_catalog_cache_for_tests()


def _chat_profile():
    return {
        "input_modalities": ["text"],
        "output_modalities": ["text"],
        "api": "chat",
    }


def test_stale_catalog_used_when_refresh_fails():
    good = {"qwen3-8-27": {"logical_id": "qwen3-8-27", "input_modalities": ["text"], "output_modalities": ["text"]}}

    async def _run():
        with patch("src.llm_model_validator._download_catalog", new_callable=AsyncMock) as dl:
            dl.return_value = good
            await _fetch_catalog(force=True)

        with patch(
            "src.llm_model_validator._download_catalog",
            new_callable=AsyncMock,
            side_effect=RuntimeError("timeout"),
        ):
            return await validate_model_for_profile("qwen3-8-27", _chat_profile())

    err = asyncio.run(_run())
    assert err is None


def test_fetch_failure_without_stale_returns_catalog_unavailable():
    async def _run():
        with patch(
            "src.llm_model_validator._download_catalog",
            new_callable=AsyncMock,
            side_effect=RuntimeError("timeout"),
        ):
            return await validate_model_for_profile("qwen3-8-27", _chat_profile())

    err = asyncio.run(_run())
    assert err is not None
    assert err["error"] == "catalog_unavailable"
    assert "not a missing-model error" in err["message"]


def test_unknown_model_when_catalog_loaded_but_missing():
    async def _run():
        with patch(
            "src.llm_model_validator._download_catalog",
            new_callable=AsyncMock,
            return_value={"other-model": {"logical_id": "other-model"}},
        ):
            return await validate_model_for_profile("qwen3-8-27", _chat_profile())

    err = asyncio.run(_run())
    assert err is not None
    assert err["error"] == "unknown_model"


def test_concurrent_refresh_single_flights():
    good = {"qwen3-8-27": {"logical_id": "qwen3-8-27", "input_modalities": ["text"], "output_modalities": ["text"]}}
    calls = {"n": 0}

    async def _slow_download():
        calls["n"] += 1
        await asyncio.sleep(0.05)
        return good

    async def _run():
        _catalog_cache["at"] = time.time() - 999
        _catalog_cache["by_id"] = {"old-model": {"logical_id": "old-model"}}
        with patch(
            "src.llm_model_validator._download_catalog",
            new_callable=AsyncMock,
            side_effect=_slow_download,
        ):
            await asyncio.gather(*(_fetch_catalog() for _ in range(8)))

    asyncio.run(_run())
    assert calls["n"] == 1
