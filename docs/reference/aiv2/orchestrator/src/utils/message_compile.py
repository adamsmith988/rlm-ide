"""
Continue-style chat history pruning for OpenAI-compatible message dicts.

Preserves the trailing tool chain (assistant with tool_calls + tool messages)
and drops oldest middle messages when over a character budget — same idea as
``extractToolSequence`` + prune in continuedev/core/llm/countTokens.ts.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple

log = logging.getLogger(__name__)


def _msg_chars(m: Dict[str, Any]) -> int:
    c = m.get("content")
    if c is None:
        n = 0
    elif isinstance(c, str):
        n = len(c)
    elif isinstance(c, list):
        try:
            n = len(json.dumps(c, ensure_ascii=False))
        except (TypeError, ValueError):
            n = len(str(c))
    else:
        n = len(str(c))
    if m.get("tool_calls"):
        try:
            n += len(json.dumps(m["tool_calls"], ensure_ascii=False))
        except (TypeError, ValueError):
            n += 400
    return n


def _split_leading_systems(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    sys_msgs: List[Dict[str, Any]] = []
    i = 0
    while i < len(messages) and messages[i].get("role") == "system":
        sys_msgs.append(messages[i])
        i += 1
    return sys_msgs, messages[i:]


def extract_protected_suffix(middle: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    ``middle`` = messages without leading system block.

    Returns (head, protected) where protected is the non-prunable tail:
    - tool, tool, ... + preceding assistant, or
    - single trailing user/assistant message.
    """
    if not middle:
        return [], []

    last = middle[-1]
    role = last.get("role")

    if role == "tool":
        j = len(middle) - 1
        while j >= 0 and middle[j].get("role") == "tool":
            j -= 1
        if j >= 0 and middle[j].get("role") == "assistant":
            return middle[:j], middle[j:]
        # Orphan tools — only protect the last message
        return middle[:-1], [last]

    # Trailing user or assistant (normal end of turn)
    return middle[:-1], [last]


def _latest_user_message(middle: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    for m in reversed(middle):
        if m.get("role") == "user":
            return m
    return None


def _ensure_latest_user_in_tail(
    head: List[Dict[str, Any]],
    protected: List[Dict[str, Any]],
    *,
    latest_user: Dict[str, Any] | None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Never send a request without the current user turn (preloop tool tails drop it otherwise)."""
    if latest_user is None:
        return head, protected
    head = [m for m in head if m is not latest_user]
    protected = [m for m in protected if m is not latest_user]
    if not protected:
        return head, [latest_user]
    if protected[0].get("role") == "assistant":
        return head, [latest_user] + protected
    return head, [latest_user] + protected


def compile_messages_continue_style(
    messages: List[Dict[str, Any]],
    *,
    char_budget: int,
    tools_extra_chars: int = 0,
) -> List[Dict[str, Any]]:
    """
    Prune oldest chat turns while keeping leading system block + protected tail.

    Args:
        messages: OpenAI-style dicts (role, content, tool_calls, tool_call_id, ...)
        char_budget: Max total chars for system + middle + protected (approximate).
        tools_extra_chars: Reserve for JSON tool definitions in the same request.
    """
    if not messages or char_budget <= 0:
        return messages

    sys_prefix, middle = _split_leading_systems(messages)
    if not middle:
        return messages

    latest_user = _latest_user_message(middle)
    head, protected = extract_protected_suffix(middle)
    head, protected = _ensure_latest_user_in_tail(head, protected, latest_user=latest_user)
    sys_cost = sum(_msg_chars(s) for s in sys_prefix)
    prot_cost = sum(_msg_chars(m) for m in protected)
    reserve = max(4_000, int(tools_extra_chars))
    available = char_budget - sys_cost - prot_cost - reserve
    if available <= 0:
        log.warning(
            "message_compile: budget exhausted after system+protected; "
            "keeping latest user + protected tail only"
        )
        return sys_prefix + protected

    while head and sum(_msg_chars(m) for m in head) > available:
        head.pop(0)
        while head and head[0].get("role") == "tool":
            head.pop(0)

    out = sys_prefix + head + protected
    if len(out) < len(messages):
        log.debug(
            "message_compile: pruned %s → %s messages (budget=%s)",
            len(messages),
            len(out),
            char_budget,
        )
    return out


def estimate_messages_chars(
    messages: List[Dict[str, Any]],
    *,
    tools_extra_chars: int = 0,
    char_budget: int = 120_000,
) -> Dict[str, Any]:
    """Approximate context fill for telemetry."""
    total = sum(_msg_chars(m) for m in (messages or [])) + int(tools_extra_chars)
    budget = max(1, int(char_budget))
    pct = round(100.0 * total / budget, 1)
    return {
        "chars": total,
        "budget": budget,
        "context_usage_pct": min(pct, 100.0),
        "over_budget": total > budget,
    }


__all__ = [
    "compile_messages_continue_style",
    "extract_protected_suffix",
    "estimate_messages_chars",
]
