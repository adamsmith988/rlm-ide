"""
Infer which session-tool family the user wants this turn (read / search / attach / list).

Vocabulary lives in ``tools/tool_intents.json``; which tools serve an intent is
decided from tool.json metadata in ``tools/llm_tool_selection.py``.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional

from .tools.tool_intents import intents, markers

SessionIntent = Literal["read", "search", "attach", "list", "graph", "discover", "general"]


def infer_session_intent(
    message: str,
    *,
    file_registry: Optional[List[Dict[str, Any]]] = None,
    discourse_focus: Optional[Dict[str, Any]] = None,
) -> SessionIntent:
    s = (message or "").strip().lower()
    if not s:
        return "general"

    if any(m in s for m in markers("graph")):
        return "graph"
    if any(m in s for m in markers("attach")):
        return "attach"
    if any(m in s for m in markers("list")):
        return "list"

    from .session_tool_resolve import (
        user_wants_file_discovery,
        user_wants_file_read,
        user_wants_more_file_content,
    )

    if user_wants_file_discovery(s):
        return "discover"

    unless = str(intents().get("search_unless_regex") or "")
    if any(m in s for m in markers("search")) and not (unless and re.search(unless, s, re.I)):
        return "search"

    if user_wants_file_read(s) or user_wants_more_file_content(s):
        return "read"

    if discourse_focus and isinstance(discourse_focus, dict):
        if discourse_focus.get("active_artifact_ids") and not user_wants_file_discovery(s):
            return "read"

    if file_registry and len(file_registry) == 1 and not user_wants_file_discovery(s):
        return "read"

    return "general"


def extract_graph_queries(
    user_message: str,
    *,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    discourse_focus: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Best-effort topic tokens for a graph search when the latest turn is meta-only."""
    stop = frozenset(intents().get("graph_query_stopwords") or [])
    sources: List[str] = [user_message or ""]
    if chat_history:
        for row in reversed(chat_history):
            if isinstance(row, dict) and row.get("role") == "user":
                body = str(row.get("content") or "").strip()
                if body:
                    sources.append(body)
            if len(sources) >= 6:
                break
    if discourse_focus and isinstance(discourse_focus, dict):
        for target in discourse_focus.get("recent_tool_targets") or []:
            if isinstance(target, dict):
                for key in ("name", "filename", "label"):
                    val = target.get(key)
                    if isinstance(val, str) and val.strip():
                        sources.append(val.strip())

    queries: List[str] = []
    for text in sources:
        for match in re.finditer(
            r"[\u0600-\u06FF]{3,24}|[A-Za-z][A-Za-z0-9_\-]{3,24}",
            text,
        ):
            token = match.group(0).strip()
            low = token.lower()
            if len(token) < 3 or low in stop or token in queries:
                continue
            queries.append(token)
            if len(queries) >= 5:
                return queries
    return queries


def meaningful_graph_queries(queries: List[str]) -> bool:
    """True when at least one query token is useful for a graph search preloop."""
    if not queries:
        return False
    stop = frozenset(intents().get("graph_query_meaningless") or [])
    for q in queries:
        token = (q or "").strip()
        if len(token) >= 3 and token.lower() not in stop:
            return True
    return False
