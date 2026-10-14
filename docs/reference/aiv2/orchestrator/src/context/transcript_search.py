"""
Transcript search — session_read (scope=session) helper for long chat sessions.

When history is too large to scan inline, the orchestrator can delegate to
``session_read`` with ``scope=session`` instead of re-reading every turn.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


def should_offer_transcript_search(
    *,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    history_summary: Optional[str] = None,
    min_turns: int = 25,
) -> bool:
    """True when transcript search is likely more efficient than full history."""
    turns = len(chat_history or [])
    if turns >= min_turns:
        return True
    if history_summary and len(history_summary) > 4000:
        return True
    return False


def build_transcript_search_hint(
    *,
    session_id: str,
    query: str,
    chat_history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Prompt block nudging the model to grep the workspace on long transcripts."""
    if not should_offer_transcript_search(chat_history=chat_history):
        return ""
    q = (query or "").strip()[:200]
    return (
        "LONG SESSION TRANSCRIPT:\n"
        f"- session_id=`{session_id}`\n"
        f"- For questions about earlier turns or buried facts, call "
        f"`session_read(session_id, scope=session, query=...)` before blind read loops.\n"
        + (f"- User query anchor: {q}\n" if q else "")
    )


async def search_transcript_via_session_read(
    *,
    tool_executor: Any,
    session_id: str,
    query: str,
    user_id: str,
    limit: int = 8,
) -> Dict[str, Any]:
    """
    Run session_read (scope=session) as a lightweight retrieval step.

    Returns the raw tool-runner envelope from the executor.
    """
    if not tool_executor or not session_id or not (query or "").strip():
        return {"ok": False, "error": "missing_args"}
    args = {
        "session_id": session_id,
        "query": query.strip(),
        "scope": "session",
        "content_scope": "all",
        "limit": limit,
    }
    try:
        result = await tool_executor.execute(
            tool_name="session_read",
            arguments=args,
            user_id=user_id,
            session_id=session_id,
        )
        return result if isinstance(result, dict) else {"ok": False, "error": "invalid_result"}
    except Exception as exc:
        log.warning("transcript search via session_read failed: %s", exc)
        return {"ok": False, "error": str(exc)}


# Backward-compatible alias
search_transcript_via_session_search = search_transcript_via_session_read

__all__ = [
    "build_transcript_search_hint",
    "search_transcript_via_session_read",
    "search_transcript_via_session_search",
    "should_offer_transcript_search",
]
