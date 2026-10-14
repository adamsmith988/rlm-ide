"""Tests for tool argument repair and validation."""
from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from services.orchestrator.src.tools.tool_arg_repair import (  # noqa: E402
    repair_tool_arguments,
    validate_tool_arguments,
)


def test_repair_session_search_workspace_wrong_key():
    raw = {"session_search_workspace": '["جعفر"]'}
    fixed = repair_tool_arguments("session_search_workspace", raw)
    assert fixed.get("query") == "جعفر"


def test_repair_graph_search_string_queries():
    fixed = repair_tool_arguments("graph_search", {"graph_search": "جعفر"})
    assert fixed.get("queries") == ["جعفر"]


def test_validate_session_search_requires_query():
    err = validate_tool_arguments("session_search_workspace", {})
    assert err is not None
    assert "query" in err


def test_validate_ok_after_repair():
    fixed = repair_tool_arguments(
        "session_search_workspace",
        {"session_search_workspace": "token"},
    )
    assert validate_tool_arguments("session_search_workspace", fixed) is None
