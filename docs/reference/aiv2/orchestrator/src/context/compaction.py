"""
Context compaction — Continue/Cline-style checkpoint summaries when context is full.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from ..config import settings
from ..utils.message_compile import estimate_messages_chars

log = logging.getLogger(__name__)

COMPACT_SUMMARY_PROMPT = """Summarize this conversation for continuation. Use this exact structure:

1. Primary Request and Intent
2. Task Evolution (original task + modifications with user quotes)
3. Current Active Task (what user most recently asked)
4. Active Files (artifact_id | filename | role in discussion)
5. Key Decisions and Tool Results
6. Pending Tasks
7. Next Step (with verbatim quote from latest user message)

Keep file artifact_ids exact. Preserve the user's language for quotes."""


async def generate_conversation_summary(
    messages: List[Dict[str, Any]],
    *,
    llm_client: Any,
    active_focus: Optional[Dict[str, Any]] = None,
) -> str:
    """LLM summary for compaction checkpoint."""
    if not llm_client or not messages:
        return ""
    transcript_lines: List[str] = []
    for m in messages[-24:]:
        if not isinstance(m, dict):
            continue
        role = m.get("role") or "?"
        content = m.get("content")
        if isinstance(content, list):
            content = " ".join(
                str(p.get("text") or "") for p in content if isinstance(p, dict)
            )
        text = str(content or "")[:1200]
        if text.strip():
            transcript_lines.append(f"{role}: {text}")
    focus_note = ""
    if isinstance(active_focus, dict) and active_focus.get("active_artifact_ids"):
        focus_note = f"\nActive focus artifact_ids: {active_focus.get('active_artifact_ids')}"
    user_body = "\n".join(transcript_lines)[-12000:] + focus_note
    try:
        result = await llm_client.chat(
            messages=[
                {"role": "system", "content": COMPACT_SUMMARY_PROMPT},
                {"role": "user", "content": user_body},
            ],
            temperature=0.1,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        if isinstance(result, dict) and result.get("ok"):
            return str(result.get("content") or "").strip()
    except Exception as exc:
        log.warning("compaction summary failed: %s", exc)
    return ""


def apply_checkpoint_to_messages(
    messages: List[Dict[str, Any]],
    summary: str,
) -> List[Dict[str, Any]]:
    """Replace middle history with summary system block + recent tail."""
    if not summary or not messages:
        return messages
    sys_msgs: List[Dict[str, Any]] = []
    rest: List[Dict[str, Any]] = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "system":
            sys_msgs.append(m)
        else:
            rest.append(m)
    recent = rest[-8:] if len(rest) > 8 else rest
    checkpoint = {
        "role": "system",
        "content": (
            "[Conversation Summary — earlier turns compacted]\n"
            f"{summary}\n\n"
            "Continue from this summary and the recent messages below. "
            "Do not restart the task from scratch."
        ),
    }
    return sys_msgs + [checkpoint] + recent


async def maybe_compact_messages(
    messages: List[Dict[str, Any]],
    *,
    llm_client: Any,
    openai_tools: Optional[List[Dict[str, Any]]] = None,
    active_focus: Optional[Dict[str, Any]] = None,
    threshold_pct: float = 80.0,
) -> Tuple[List[Dict[str, Any]], Optional[str], Dict[str, Any]]:
    """
    Auto-compact when context exceeds threshold.

    Returns (messages, summary_or_none, telemetry).
    """
    tools_chars = len(str(openai_tools or [])) // 4
    telem = estimate_messages_chars(
        messages,
        tools_extra_chars=tools_chars,
        char_budget=int(settings.STREAM_CONTEXT_CHAR_BUDGET),
    )
    if telem.get("context_usage_pct", 0) < threshold_pct:
        return messages, None, telem

    log.info(
        "context compaction triggered at %.1f%% (%s chars)",
        telem.get("context_usage_pct"),
        telem.get("chars"),
    )
    summary = await generate_conversation_summary(
        messages, llm_client=llm_client, active_focus=active_focus
    )
    if not summary:
        return messages, None, telem
    compacted = apply_checkpoint_to_messages(messages, summary)
    new_telem = estimate_messages_chars(
        compacted,
        tools_extra_chars=tools_chars,
        char_budget=int(settings.STREAM_CONTEXT_CHAR_BUDGET),
    )
    return compacted, summary, new_telem


__all__ = [
    "apply_checkpoint_to_messages",
    "generate_conversation_summary",
    "maybe_compact_messages",
]
