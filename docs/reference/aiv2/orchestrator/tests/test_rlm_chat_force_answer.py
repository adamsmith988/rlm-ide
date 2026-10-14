"""Hub chat must force a user-facing answer after tools (not idle empty)."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest


def _extract(resp: Dict[str, Any]) -> Dict[str, Any]:
    choices = resp.get("choices") or []
    if choices and isinstance(choices[0], dict):
        msg = choices[0].get("message") or {}
        if isinstance(msg, dict):
            return dict(msg)
    return {"role": "assistant", "content": str(resp.get("content") or "")}


async def _noop_tool(*_a, **_k):
    return {"ok": True}


@pytest.mark.asyncio
async def test_chat_lifetime_forces_answer_after_empty_tools(tmp_path):
    """Model tool-calls then ghosts → hub chat must nudge a final prose answer."""
    from src.orchestrators.rlm.runtime.session import RlmSession

    calls: List[str] = []

    async def chat_fn(body: Dict[str, Any]) -> Dict[str, Any]:
        n = len(calls)
        calls.append("x")
        # First turn: ipython only (no prose) — classic hub-empty failure.
        if n == 0:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "ipython",
                                        "arguments": '{"code":"print(1)"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        # Later turns (synthesis nudge): real user-facing answer.
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "نتیجه: شخصیت برجسته‌ای در این نمونه پیدا نشد.",
                    }
                }
            ]
        }

    async def run_ipython(self, code: str) -> str:
        return "Keyword نام: []\nKeyword شخصیت: []"

    sess = RlmSession(
        cfg={
            "model": "m",
            "rlm_max_turns": 8,
            "tool_ids": [],
            "run_lifetime": "chat",
            "rlm_autonomous_max_continuations": 0,
        },
        user_content="تحلیل شخصیت‌ها",
        artifact_dir=str(tmp_path / "chat_synth"),
        chat_fn=chat_fn,
        extract_assistant_fn=_extract,
        gen_params={},
        tool_runner=_noop_tool,
    )

    async def skip_kernel() -> None:
        return None

    sess.start_kernel = skip_kernel  # type: ignore[method-assign]
    sess._run_ipython = run_ipython.__get__(sess, RlmSession)  # type: ignore[method-assign]

    out = await sess.continue_with_message("تحلیل شخصیت‌ها")
    assert "شخصیت" in str(out or sess.last_content or "")
    assert str(sess.last_content or "").strip()
    assert sess.status == "idle"
    # Synthesis must call the model again after the tool turn.
    assert len(calls) >= 2
