"""
Session task checklist — lightweight focus anchor for long conversations.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_task_checklist_block(
    *,
    user_message: str,
    pinned_facts: Optional[Dict[str, Any]] = None,
    discourse_focus: Optional[Dict[str, Any]] = None,
    checklist_state: Optional[Dict[str, Any]] = None,
) -> str:
    """Render a short checklist block for the system prompt."""
    lines: List[str] = ["SESSION TASK CHECKLIST:"]
    goal = ""
    if isinstance(pinned_facts, dict):
        goal = str(pinned_facts.get("goal") or pinned_facts.get("primary_goal") or "").strip()
    if goal:
        lines.append(f"- Goal: {goal[:400]}")
    lines.append(f"- Latest user message: {user_message[:300]}")
    if isinstance(discourse_focus, dict):
        aids = discourse_focus.get("active_artifact_ids") or []
        if aids:
            lines.append(f"- Active file focus: {', '.join(str(a) for a in aids[:2])}")
        reason = discourse_focus.get("focus_reason")
        if reason:
            lines.append(f"- Focus reason: {reason}")
    done = (checklist_state or {}).get("done") or []
    pending = (checklist_state or {}).get("pending") or []
    if done:
        lines.append("- Done: " + "; ".join(str(x) for x in done[:5]))
    if pending:
        lines.append("- Pending: " + "; ".join(str(x) for x in pending[:5]))
    if len(lines) <= 2:
        return ""
    return "\n".join(lines)


def update_checklist_from_tool_run(
    state: Dict[str, Any],
    *,
    tool_name: str,
    ok: bool,
    tool_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Track simple done/pending items from tool activity."""
    out = dict(state or {})
    done: List[str] = list(out.get("done") or [])
    label = tool_name
    if isinstance(tool_args, dict):
        fid = tool_args.get("file_id") or tool_args.get("filename")
        if fid:
            label = f"{tool_name}({fid})"
    entry = f"{label}: {'ok' if ok else 'failed'}"
    if entry not in done:
        done.append(entry)
    out["done"] = done[-12:]
    return out


__all__ = ["build_task_checklist_block", "update_checklist_from_tool_run"]
