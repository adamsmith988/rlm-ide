"""Regression: flattened tool results must not dump ambiguous JSON into chat."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

_SRC = Path(__file__).resolve().parents[1] / "src"


def _exec_helpers(*names: str) -> Dict[str, Any]:
    """Load selected function definitions from source without importing graph package."""
    src = (_SRC / "graph" / "_archived" / "cline_engine.py").read_text(encoding="utf-8")
    native = (_SRC / "graph" / "native_tools_engine.py").read_text(encoding="utf-8")
    ns: Dict[str, Any] = {
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "json": __import__("json"),
    }
    chunks = {
        "_extract_llm_summary": (src, "def _extract_llm_summary", "\ndef _format_tool_result_message"),
        "_tool_result_payload": (
            native,
            "def _tool_result_payload",
            "\ndef _is_ambiguous_attach_result",
        ),
        "_ambiguous_attach_user_answer": (
            native,
            "def _ambiguous_attach_user_answer",
            "\ndef _deterministic_answer_from_tool_runs",
        ),
    }
    for name in names:
        blob, start_marker, end_marker = chunks[name]
        start = blob.index(start_marker)
        end = blob.index(end_marker, start)
        exec(blob[start:end], ns)
    return ns


def test_extract_llm_summary_prefers_top_level_channels():
    ns = _exec_helpers("_extract_llm_summary")
    result = {
        "ok": True,
        "data": {
            "status": "ambiguous",
            "count": 2,
            "candidates": [{"filename": "a.json"}, {"filename": "b.json"}],
        },
        "channels": {
            "llm": "⚠️ Multiple files match (2). Nothing was attached yet.\n1. `a.json`",
            "ui": {"title": "Multiple files match", "candidates": []},
        },
        "tool_id": "session_attach",
    }
    text = ns["_extract_llm_summary"](result)
    assert text.startswith("⚠️ Multiple files match")
    assert '"status": "ambiguous"' not in text


def test_ambiguous_attach_user_answer_lists_filenames_only():
    ns = _exec_helpers("_tool_result_payload", "_ambiguous_attach_user_answer")
    result = {
        "ok": True,
        "data": {
            "status": "ambiguous",
            "count": 2,
            "candidates": [
                {
                    "artifact_id": "id-1",
                    "filename": "work_entries.json",
                    "path": "minio://uploads/users/x/work_entries.json",
                    "case_id": "cas_1",
                    "media_type": "document",
                },
                {
                    "artifact_id": "id-2",
                    "filename": "work_entries.json",
                    "path": "minio://uploads/users/y/work_entries.json",
                },
            ],
        },
        "channels": {"llm": "ignored for user answer"},
    }
    text = ns["_ambiguous_attach_user_answer"](result, fallback='{"status":"ambiguous"}')
    assert "Multiple files match (2)" in text
    assert "work_entries.json" in text
    assert "minio://" not in text
    assert '"status"' not in text
