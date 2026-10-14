"""Prompt must not duplicate tool catalogs; tools[] is the single source."""
from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pathlib import Path

from services.orchestrator.src.prompts.qwen3_prompts import (  # noqa: E402
    QWEN3_SYSTEM_PROMPT,
    build_qwen3_system_messages,
)

def test_qwen3_system_prompt_has_no_dynamic_tools_placeholder():
    assert "{dynamic_tools_info}" not in QWEN3_SYSTEM_PROMPT
    assert "Available Tools" not in QWEN3_SYSTEM_PROMPT
    assert "single source" in QWEN3_SYSTEM_PROMPT.lower() or "only" in QWEN3_SYSTEM_PROMPT


def test_build_qwen3_system_messages_excludes_tool_catalog_blocks():
    messages = build_qwen3_system_messages(chat_session_id="sess-test-uuid")
    system_text = "\n\n".join(
        str(m.get("content") or "")
        for m in messages
        if m.get("role") == "system"
    )
    assert "SESSION / FILE TOOLS" not in system_text
    assert "ACTIVE TOOLS THIS TURN" not in system_text
    assert "File-Based Tools" not in system_text
    assert "session_read_file —" not in system_text


def test_native_engine_source_excludes_tool_catalog_in_prompt():
    """Avoid importing native_tools_engine (LangGraph deps); verify source contract."""
    root = Path(__file__).resolve().parents[1]
    src = (root / "src" / "graph" / "native_tools_engine.py").read_text(encoding="utf-8")
    assert "_format_active_tools_block" not in src
    assert "SESSION_TOOLS_DECISION_BLOCK" not in src
    assert "ACTIVE TOOLS THIS TURN" not in src
    assert "catalog lives in vLLM" in src or "tools`` request parameter" in src


def test_openai_tools_carry_descriptions():
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": "session_files",
                "description": "List files in this chat workspace.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    fn = openai_tools[0]["function"]
    assert fn["name"] == "session_files"
    assert len(fn.get("description", "")) > 10
