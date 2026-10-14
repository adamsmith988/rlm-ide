"""Regression: unlimited chat output helpers (tool text cap, no soft 4k)."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"


def _load_cline_cap():
    path = _SRC / "graph" / "_archived" / "cline_engine.py"
    src = path.read_text(encoding="utf-8")
    start = src.index("def _cap_llm_text")
    end = src.index("\ndef _normalized_tool_summary", start)
    ns: dict = {"Optional": type(None)}
    # Minimal settings stub
    class _S:
        STREAM_TOOL_RESULT_MAX_CHARS = 24_000
    ns["settings"] = _S()
    exec(src[start:end], ns)
    return ns["_cap_llm_text"]


def test_cap_llm_text_uses_stream_tool_result_max_chars():
    cap = _load_cline_cap()
    long = "x" * 30_000
    out = cap(long)
    assert len(out) == 24_000
    assert out == long[:24_000]


def test_cap_llm_text_passthrough_under_limit():
    cap = _load_cline_cap()
    assert cap("hello") == "hello"


def test_vllm_chat_stream_has_no_repetition_abort_break():
    """chat_stream_with_tools must not break mid-stream on repetition."""
    path = _SRC / "llm" / "gateway_llm_client.py"
    src = path.read_text(encoding="utf-8")
    # The mid-stream abort block was removed from chat_stream_with_tools.
    assert "Aborting stream: repetitive content loop detected" not in src
    assert 'yield {\n                    "type": "repetition_aborted"' not in src
