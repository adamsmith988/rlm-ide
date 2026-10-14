"""Gateway 429 handling in GatewayLLMClient and newest-page history fetch."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import httpx
import pytest
from openai import RateLimitError

from src.chat_history import ChatHistoryClient
from src.config import settings
from src.llm import gateway_llm_client as glc


def _rate_limit(retry_after: str | None = None) -> RateLimitError:
    headers = {"retry-after": retry_after} if retry_after else {}
    req = httpx.Request("POST", "http://gw/v1/chat/completions")
    resp = httpx.Response(429, headers=headers, request=req, json={"detail": "rate_limit_exceeded"})
    return RateLimitError("rate_limit_exceeded", response=resp, body=None)


class _Completions:
    def __init__(self, outcomes: List[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    async def create(self, **_params: Any) -> Any:
        self.calls += 1
        out = self.outcomes.pop(0)
        if isinstance(out, BaseException):
            raise out
        return out


def _client(outcomes: List[Any]) -> tuple[glc.GatewayLLMClient, _Completions]:
    client = glc.GatewayLLMClient.__new__(glc.GatewayLLMClient)
    completions = _Completions(outcomes)
    client._client = type("C", (), {"chat": type("Ch", (), {"completions": completions})()})()
    return client, completions


@pytest.fixture
def sleeps(monkeypatch):
    waits: List[float] = []

    async def fake_sleep(sec: float) -> None:
        waits.append(sec)

    monkeypatch.setattr(glc.asyncio, "sleep", fake_sleep)
    return waits


def test_rate_limit_backs_off_exponentially_then_succeeds(sleeps):
    client, comp = _client([_rate_limit(), _rate_limit(), "ok"])
    assert asyncio.run(client.create_completion(model="m")) == "ok"
    base = float(settings.LLM_RATE_LIMIT_BACKOFF_SEC)
    assert sleeps == [base, base * 2]
    assert comp.calls == 3


def test_rate_limit_honours_retry_after(sleeps):
    client, _ = _client([_rate_limit("7"), "ok"])
    assert asyncio.run(client.create_completion(model="m")) == "ok"
    assert sleeps == [7.0]


def test_rate_limit_gives_up_after_wait_budget(sleeps, monkeypatch):
    monkeypatch.setattr(settings, "LLM_RATE_LIMIT_MAX_WAIT_SEC", 5.0)
    client, _ = _client([_rate_limit("4"), _rate_limit("4"), "ok"])
    with pytest.raises(RateLimitError):
        asyncio.run(client.create_completion(model="m"))
    assert sleeps == [4.0]


def test_get_messages_returns_newest_rows_across_pages(monkeypatch):
    monkeypatch.setattr(settings, "STORAGE_MESSAGES_PAGE_SIZE", 4)
    rows = [{"id": i} for i in range(10)]
    seen: List[Dict[str, Any]] = []

    async def fake_request(self, method, endpoint, params=None, **_kw):
        seen.append(dict(params or {}))
        off, lim = params["offset"], params["limit"]
        return rows[off: off + lim]

    monkeypatch.setattr(ChatHistoryClient, "_request", fake_request)
    client = ChatHistoryClient.__new__(ChatHistoryClient)
    out = asyncio.run(client.get_messages("s", limit=3))
    assert [r["id"] for r in out] == [7, 8, 9]
    assert [p["offset"] for p in seen] == [0, 4, 8]


def test_get_messages_short_session_single_page(monkeypatch):
    monkeypatch.setattr(settings, "STORAGE_MESSAGES_PAGE_SIZE", 500)

    async def fake_request(self, method, endpoint, params=None, **_kw):
        return [{"id": 1}, {"id": 2}]

    monkeypatch.setattr(ChatHistoryClient, "_request", fake_request)
    client = ChatHistoryClient.__new__(ChatHistoryClient)
    assert [r["id"] for r in asyncio.run(client.get_messages("s", limit=20))] == [1, 2]
