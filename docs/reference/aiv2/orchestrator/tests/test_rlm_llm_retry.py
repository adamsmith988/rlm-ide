"""Hub RLM LLM reconnect — Agent Flow parity (503 / transport)."""

from __future__ import annotations

import asyncio

import pytest

from src.orchestrators.host_adapters.llm_retry import (
    LLM_503_RETRY_ATTEMPTS,
    LLM_503_RETRY_DELAY_SEC,
    LLM_TRANSPORT_RETRY_DELAY_SEC,
    call_with_llm_retry,
    is_retryable_llm_503,
    is_retryable_llm_transport,
    llm_retry_plan,
    notify_llm_reconnecting,
    reconnect_preview,
)
from src.orchestrators.host_adapters.hub_sse import HubEventBridge


def test_hub_retryable_all_backends_failed_without_af_prefix():
    """OpenAI client surfaces bare proxy aggregate errors — still 503 plan."""
    err = RuntimeError(
        "all_backends_failed: openai_compat:osbi#cred_x:ConnectError:"
        "[Errno -3] Temporary failure in name resolution"
    )
    assert is_retryable_llm_503(err)
    assert not is_retryable_llm_transport(err)  # 503 plan owns it
    plan = llm_retry_plan(err)
    assert plan is not None
    assert plan[0] == LLM_503_RETRY_ATTEMPTS
    assert plan[1] == LLM_503_RETRY_DELAY_SEC
    assert plan[2] == "model gateway 503"


def test_hub_retryable_catalog_and_af_prefixed_503():
    assert is_retryable_llm_503(
        RuntimeError(
            "LLM error 503: LLM gateway catalog is temporarily unavailable; "
            "retry shortly (this is not a missing-model error)."
        )
    )
    assert is_retryable_llm_503(
        RuntimeError("LLM error 503: all_backends_failed: openai_compat:n1")
    )


def test_hub_not_retryable_overflow_or_413():
    assert not is_retryable_llm_503(
        RuntimeError(
            "LLM error 503: maximum context length is 8192 tokens. "
            "Your request had 9000 input tokens"
        )
    )
    assert not is_retryable_llm_503(
        RuntimeError(
            "all_backends_failed: openai_compat:osbi:"
            "Client error '413 Request Entity Too Large' for url "
            "'https://example.test/v1/chat/completions'"
        )
    )
    assert llm_retry_plan(RuntimeError("LLM error 400: unknown_model")) is None


def test_hub_transport_readerror_plan():
    err = RuntimeError(
        "LLM stream failed ReadError: no message "
        "url=http://llm-gateway:8005/v1/chat/completions"
    )
    assert is_retryable_llm_transport(err)
    plan = llm_retry_plan(err)
    assert plan is not None
    assert plan[2] == "LLM transport"
    assert plan[1] == LLM_TRANSPORT_RETRY_DELAY_SEC


@pytest.mark.asyncio
async def test_call_with_llm_retry_then_success(monkeypatch):
    calls = {"n": 0}
    sleeps: list[float] = []
    notes: list[str] = []

    async def boom_then_ok():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError(
                "all_backends_failed: openai_compat:osbi:ConnectError: dns"
            )
        return {"ok": True, "choices": []}

    async def fake_sleep(sec):
        sleeps.append(sec)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    def _note(*, failed_attempt, max_attempts, wait_sec, kind, exc=None, **_):
        del exc
        notes.append(
            reconnect_preview(
                failed_attempt=failed_attempt,
                max_attempts=max_attempts,
                wait_sec=wait_sec,
                kind=kind,
            )
        )

    out = await call_with_llm_retry(boom_then_ok, on_retry=_note)
    assert out["ok"] is True
    assert calls["n"] == 3
    assert sleeps == [LLM_503_RETRY_DELAY_SEC, LLM_503_RETRY_DELAY_SEC]
    assert len(notes) == 2
    assert "reconnecting" in notes[0]


@pytest.mark.asyncio
async def test_call_with_llm_retry_exhausted(monkeypatch):
    calls = {"n": 0}

    async def always_fail():
        calls["n"] += 1
        raise RuntimeError("chat_model_unavailable: no healthy backends")

    async def fake_sleep(_sec):
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    with pytest.raises(RuntimeError, match="chat_model_unavailable"):
        await call_with_llm_retry(always_fail)
    assert calls["n"] == LLM_503_RETRY_ATTEMPTS


def test_hub_sse_maps_llm_reconnect():
    bridge = HubEventBridge()
    events = bridge.map_event(
        {
            "kind": "llm_reconnect",
            "preview": "model gateway 503 — reconnecting, retry 2/12 in 10s",
        }
    )
    types = [e.get("type") for e in events]
    assert "thinking_start" in types
    thinking = next(e for e in events if e.get("type") == "thinking")
    assert "[reconnecting]" in thinking["content"]
    assert "retry 2/12" in thinking["content"]


def test_notify_llm_reconnecting_emits_on_event():
    seen: list[dict] = []
    preview = notify_llm_reconnecting(
        failed_attempt=1,
        max_attempts=12,
        wait_sec=10.0,
        kind="model gateway 503",
        on_event=seen.append,
    )
    assert "reconnecting" in preview
    assert len(seen) == 1
    assert seen[0]["kind"] == "llm_reconnect"
    assert seen[0]["attempt"] == 2
