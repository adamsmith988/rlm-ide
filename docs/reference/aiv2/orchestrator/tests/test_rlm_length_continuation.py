"""Length-stitch + AF-style truncated tool-call recovery."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from src.orchestrators.rlm.runtime.chat_stream import (
    TRUNCATED_TOOL_ARGS_TOOL_RESULT,
    build_length_continue_user,
    completion_finish_reason,
    invoke_with_length_continuation,
    length_continuation_body,
    should_continue_on_length,
    should_retry_truncated_tool_calls,
    stitch_length_completions,
)


def _completion(content: str, finish: str, *, tools: bool = False) -> Dict[str, Any]:
    msg: Dict[str, Any] = {"role": "assistant", "content": content}
    if tools:
        msg["tool_calls"] = [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "ipython", "arguments": '{"code":"x"},'},
            }
        ]
    return {
        "choices": [
            {"index": 0, "message": msg, "finish_reason": finish},
        ]
    }


def test_should_continue_on_length_only_for_prose():
    assert should_continue_on_length(_completion("hello", "length"))
    assert not should_continue_on_length(_completion("hello", "stop"))
    assert not should_continue_on_length(_completion("", "length"))
    assert not should_continue_on_length(_completion("x", "length", tools=True))


def test_should_continue_on_reasoning_only_length():
    data = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "still thinking about the report…",
                },
            }
        ]
    }
    assert should_continue_on_length(data)


def test_should_continue_when_finish_missing_but_usage_hits_cap():
    data = {
        "choices": [
            {
                "finish_reason": None,
                "message": {"role": "assistant", "content": "partial answer"},
            }
        ],
        "usage": {"completion_tokens": 1000},
    }
    assert should_continue_on_length(data, max_tokens=1000)
    assert not should_continue_on_length(data, max_tokens=4000)


def test_should_continue_when_proxy_lies_with_stop_but_usage_hits_cap():
    # Incomplete prose (no sentence end) + usage at cap → continue.
    data = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "partial answer mid"},
            }
        ],
        "usage": {"completion_tokens": 750},
    }
    assert should_continue_on_length(data, max_tokens=750)


def test_should_not_continue_when_stop_with_complete_closing():
    """Natural stop after a finished list must not stitch forever."""
    closing = "لیست به پایان رسید. تمامی ۲۵۰ فایل با موفقیت فهرست شدند."
    data = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": ("file-1\n" * 80) + closing,
                },
            }
        ],
        "usage": {"completion_tokens": 750},
    }
    assert not should_continue_on_length(data, max_tokens=750)


def test_should_not_continue_when_stop_but_chars_near_token_budget():
    """Char≈2×max_tokens alone must not continue when finish_reason=stop."""
    data = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": ("x" * 500) + "."},
            }
        ]
    }
    assert not should_continue_on_length(data, max_tokens=200)
    assert not should_continue_on_length(data, max_tokens=8000)


def test_should_continue_unclosed_channel():
    data = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "<|channel>thought\nplanning the file",
                },
            }
        ]
    }
    assert should_continue_on_length(data)


def test_stitch_merges_content_and_last_finish():
    a = _completion("Part one ", "length")
    b = _completion("part two", "stop")
    merged = stitch_length_completions(a, b)
    assert completion_finish_reason(merged) == "stop"
    assert merged["choices"][0]["message"]["content"] == "Part one part two"


def test_stitch_strips_restarted_introduction():
    prev = "Title\n\nParagraph one about cats.\n\nParagraph two continues"
    restarted = (
        "Title\n\nParagraph one about cats.\n\nParagraph two continues "
        "with the missing ending."
    )
    merged = stitch_length_completions(
        _completion(prev, "length"),
        _completion(restarted, "stop"),
    )
    out = merged["choices"][0]["message"]["content"]
    assert out.count("Title") == 1
    assert out.endswith("with the missing ending.")


def test_continue_user_quotes_stopping_point():
    partial = "A" * 100 + "STOP_MARKER_HERE exact cut"
    nudge = build_length_continue_user(partial, anchor_chars=40)
    assert "ALREADY_WRITTEN_ENDS_HERE" in nudge
    assert "CONTINUE_AFTER_THIS" in nudge
    assert "STOP_MARKER_HERE exact cut" in nudge
    assert "Resume EXACTLY after" in nudge


def test_length_continuation_body_embeds_anchor():
    body = {
        "messages": [{"role": "user", "content": "write a long report"}],
        "tools": [{"type": "function"}],
        "max_tokens": 128,
    }
    partial = _completion("Intro…\n\nSection 1 ends mid-sente", "length")
    req = length_continuation_body(body, partial)
    assert "tools" not in req
    user = req["messages"][-1]
    assert user["role"] == "user"
    assert "ALREADY_WRITTEN_ENDS_HERE" in user["content"]
    assert "mid-sente" in user["content"]
    assert req["messages"][-2]["role"] == "assistant"


@pytest.mark.asyncio
async def test_invoke_stitches_until_stop():
    calls: List[Dict[str, Any]] = []

    async def fake_invoke(body: Dict[str, Any], on_delta=None) -> Dict[str, Any]:
        calls.append(body)
        n = len(calls)
        if n == 1:
            if on_delta:
                on_delta({"content": "AAA"})
            return _completion("AAA", "length")
        if n == 2:
            if on_delta:
                on_delta({"content": "BBB"})
            return _completion("BBB", "length")
        if on_delta:
            on_delta({"content": "CCC"})
        return _completion("CCC", "stop")

    deltas: List[str] = []
    out = await invoke_with_length_continuation(
        fake_invoke,
        {"messages": [{"role": "user", "content": "write a lot"}], "max_tokens": 8},
        on_delta=lambda d: deltas.append(str(d.get("content") or "")),
        max_continuations=8,
    )
    assert len(calls) == 3
    assert out["choices"][0]["message"]["content"] == "AAABBBCCC"
    assert completion_finish_reason(out) == "stop"
    assert "".join(deltas) == "AAABBBCCC"
    # Continuations drop tools and carry an anchored continue nudge.
    assert "tools" not in calls[1]
    assert any(
        "ALREADY_WRITTEN_ENDS_HERE" in str(m.get("content") or "")
        or "cut off" in str(m.get("content") or "")
        for m in calls[1].get("messages") or []
        if isinstance(m, dict)
    )


@pytest.mark.asyncio
async def test_invoke_respects_zero_continuations():
    async def fake_invoke(body: Dict[str, Any], on_delta=None) -> Dict[str, Any]:
        return _completion("cut", "length")

    out = await invoke_with_length_continuation(
        fake_invoke,
        {"messages": [{"role": "user", "content": "x"}]},
        max_continuations=0,
    )
    assert out["choices"][0]["message"]["content"] == "cut"
    assert completion_finish_reason(out) == "length"


def test_should_retry_truncated_tool_calls():
    bad = _completion("", "length", tools=True)
    # force truncated args like AF test
    bad["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (
        '{"code": "print(1'
    )
    assert should_retry_truncated_tool_calls(bad)
    assert not should_continue_on_length(bad)
    good = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "ipython",
                                "arguments": '{"code": "print(1)"}',
                            },
                        }
                    ],
                },
            }
        ]
    }
    assert not should_retry_truncated_tool_calls(good)


@pytest.mark.asyncio
async def test_invoke_retries_truncated_tool_then_returns_valid():
    calls: List[Dict[str, Any]] = []
    retries: List[int] = []

    async def fake_invoke(body: Dict[str, Any], on_delta=None) -> Dict[str, Any]:
        calls.append(body)
        if len(calls) == 1:
            return {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "tc_bad",
                                    "type": "function",
                                    "function": {
                                        "name": "ipython",
                                        "arguments": '{"code": "print(1',
                                    },
                                }
                            ],
                        },
                    }
                ]
            }
        # Retry request must carry AF tool-result notice and keep tools.
        msgs = body.get("messages") or []
        assert any(
            isinstance(m, dict)
            and m.get("role") == "tool"
            and TRUNCATED_TOOL_ARGS_TOOL_RESULT in str(m.get("content") or "")
            for m in msgs
        )
        assert body.get("tools") == [{"type": "function"}]
        return {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "tc_ok",
                                "type": "function",
                                "function": {
                                    "name": "ipython",
                                    "arguments": '{"code": "print(1)"}',
                                },
                            }
                        ],
                    },
                }
            ]
        }

    out = await invoke_with_length_continuation(
        fake_invoke,
        {
            "messages": [{"role": "user", "content": "run"}],
            "tools": [{"type": "function"}],
        },
        max_tool_retries=2,
        on_tool_retry=lambda n, _p: retries.append(n),
    )
    assert len(calls) == 2
    assert retries == [1]
    assert not should_retry_truncated_tool_calls(out)
    args = out["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    assert args == '{"code": "print(1)"}'
